import clip
import torch
import torch.nn.functional as F

from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
)
from fs.utils.eval_utils import clip_classifier, cls_acc, evaluate


def run_lora(
    args,
    clip_model,
    logit_scale,
    dataset,
    train_loader,
    val_loader,
    test_loader,
):
    validate = getattr(args, "validate", False)

    # textual features of the training set
    textual_features = clip_classifier(
        dataset.classnames, dataset.template, clip_model
    )

    # plug LoRA layers
    list_lora_layers = apply_lora(args, clip_model, verbose=False)
    mark_only_lora_as_trainable(clip_model)
    trainable_params = get_lora_parameters(clip_model)

    clip_model = clip_model.cuda().float()
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-6
    )

    # training LoRA
    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0

    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0
        if args.encoder == "vision":
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(train_loader):
            template = dataset.template[0]
            texts = [
                template.format(classname.replace("_", " "))
                for classname in dataset.classnames
            ]
            images, target = images.cuda(), target.cuda()

            if args.encoder == "text" or args.encoder == "both":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    tokenized = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(tokenized)
                text_features = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )

            if args.encoder == "vision" or args.encoder == "both":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

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
            print(
                "[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                    count_iters, total_iters, current_lr, acc_train, loss_epoch
                )
            )

        # Eval
        if validate:
            clip_model.eval()
            acc_val = evaluate(
                clip_model,
                val_loader,
                template=dataset.template[0],
                classnames=dataset.val_classnames,
                label_to_expert=None,
            )
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    if args.save_path is not None:
        save_lora(args, list_lora_layers)

    # evaluate on test sets after training
    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader

        # evaluation on base classes
        acc_test_base = evaluate(
            clip_model,
            test_base_loader,
            template=dataset.template[0],
            classnames=dataset.test_classnames,
            label_to_expert=None,
        )
        print("**** Test-Base accuracy: {:.2f}. ****\n".format(acc_test_base))

        # evaluation on novel classes
        acc_test_novel = evaluate(
            clip_model,
            test_new_loader,
            template=dataset.template[0],
            classnames=dataset.test_new_classnames,
            label_to_expert=None,
            use_expert=False,
        )
        print("**** Test-Novel accuracy: {:.2f}. ****\n".format(acc_test_novel))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_novel}

    else:
        acc_test = evaluate(
            clip_model,
            test_loader,
            template=dataset.template[0],
            classnames=dataset.test_classnames,
            label_to_expert=None,
        )
        print(
            "\n**** Final test accuracy (all categories): {:.2f}. ****\n".format(
                acc_test
            )
        )
        result = {"acc_test": acc_test}

    return result
