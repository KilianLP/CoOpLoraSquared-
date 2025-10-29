import clip
import torch
import torch.nn.functional as F

from loralib import layers as lora_layers
from loralib.utils import (
    mark_only_lora_as_trainable, apply_lora, get_lora_parameters, 
    lora_state_dict, save_lora, load_lora
)
from lorasquaredlib import (
    apply_lorasquared,
    mark_only_lorasquared_as_trainable,
    get_lorasquared_parameters,
    resolve_expert_indices,
)
from fs.utils.eval_utils import clip_classifier, cls_acc, evaluate


def _set_active_expert(layers, expert_selection):
    if layers is None:
        return
    for layer in layers:
        if hasattr(layer, "set_active_expert"):
            layer.set_active_expert(expert_selection)


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False

    # textual features of the training set
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    
    active_expert = None

    # plug LoRA layers
    if args.lora_variant == 'squared':
        list_lora_layers = apply_lorasquared(
            clip_model,
            backbone=args.backbone,
            encoder=args.encoder,
            position=args.position,
            params=args.params,
            r_shared=args.lora_shared_rank,
            r_expert=args.lora_expert_rank,
            n_experts=args.lora_num_experts,
            alpha_shared=args.alpha,
            alpha_expert=args.alpha,
            dropout_rate=args.dropout_rate,
            verbose=False
        )
        active_expert = resolve_expert_indices(args.lora_active_expert, args.lora_num_experts)
        _set_active_expert(list_lora_layers, active_expert)
        mark_only_lorasquared_as_trainable(clip_model)
        trainable_params = get_lorasquared_parameters(clip_model)
    else:
        list_lora_layers = apply_lora(args, clip_model, verbose=False)
        mark_only_lora_as_trainable(clip_model)
        trainable_params = get_lora_parameters(clip_model)

    clip_model = clip_model.cuda().float()
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # training LoRA
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0
    
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        
        for i, (images, target) in enumerate(train_loader):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break

            if args.debug and count_iters >= len(train_loader):
                count_iters = total_iters
                break
            
        if count_iters <= total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(count_iters, total_iters, current_lr, acc_train, loss_epoch))

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate(clip_model, val_loader, template=dataset.template[0], classnames=dataset.val_classnames)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
    
    if args.save_path is not None:
        if args.lora_variant == 'squared':
            print("LoRA^2 saving is not implemented yet; skipping serialization.")
        else:
            save_lora(args, list_lora_layers)

    # evaluate on test sets after training
    if args.lora_variant == 'squared':
        _set_active_expert(list_lora_layers, active_expert)

    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader
        
        # evaluation on base classes
        acc_test_base = evaluate(
            clip_model, test_base_loader, template=dataset.template[0], classnames=dataset.test_classnames
        )
        print("**** Test-Base accuracy: {:.2f}. ****\n".format(acc_test_base))

        # evaluation on novel classes
        acc_test_novel = evaluate(
            clip_model, test_new_loader, template=dataset.template[0], classnames=dataset.test_new_classnames
        )
        print("**** Test-Novel accuracy: {:.2f}. ****\n".format(acc_test_novel))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_novel}
    
    else:
        acc_test = evaluate(
            clip_model, test_loader, template=dataset.template[0], classnames=dataset.test_classnames
        )
        print("\n**** Final test accuracy (all categories): {:.2f}. ****\n".format(acc_test))
        result = {"acc_test": acc_test}


    return result
            
    
            
