<div align="center">
<h2>CVPR 2025<br>Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages</h2>

<p>
  <a href="https://scholar.google.com/citations?user=SxQwDD8AAAAJ&authuser=1">Matteo Farina</a>, 
  <a href="https://scholar.google.com/citations?user=bqTPA8kAAAAJ&authuser=1">Massimiliano Mancini</a>, 
  <a href="https://scholar.google.com/citations?user=qSw6YfcAAAAJ&authuser=1">Giovanni Iacca</a> and 
  <a href="https://scholar.google.com/citations?user=xf1T870AAAAJ&authuser=1">Elisa Ricci</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2503.11609-b31b1b.svg)](https://arxiv.org/abs/2503.11609)

</div>

>**Abstract.** *An old-school recipe for training a classifier is to (i) learn a good feature extractor and (ii) optimize a linear layer atop. When only a handful of samples are available per category, as in Few-Shot Adaptation (FSA), data are insufficient to fit a large number of parameters, rendering the above impractical. This is especially true with large pre-trained Vision-Language Models (VLMs), which motivated successful research at the intersection of Parameter-Efficient Fine-tuning (PEFT) and FSA. In this work, we start by analyzing the learning dynamics of PEFT techniques when trained on few-shot data from only a subset of categories, referred to as the “base” classes. We show that such dynamics naturally splits into two distinct phases: (i) task-level feature extraction and (ii) specialization to the available concepts. To accommodate this dynamic, we then depart from prompt- or adapter-based methods and tackle FSA differently. Specifically, given a fixed computational budget, we split it to (i) learn a task-specific feature extractor via PEFT and (ii) train a linear classifier on top. We call this scheme Two-Stage Few-Shot Adaptation (2SFS). Differently from established methods, our scheme enables a novel form of selective inference at a category level, i.e., at test time, only novel categories are embedded by the adapted text encoder, while embeddings of base categories are available within the classifier. Results with fixed hyperparameters across two settings, three backbones, and eleven datasets, show that 2SFS matches or surpasses the state-of-the-art, while established methods degrade significantly across settings.*

### Updates [dd/mm/yy]
- [30/03/25] Code out! Also, check out our [effort for reproducibility in Few-Shot Learning](#an-effort-for-reproducibility-in-few-shot-learning)!
- [07/03/25] ~Code release happening soon! (ETA 2/3 weeks 😊)~


### Install Dependencies
Create a virtual environment with your preferred tools and install requirements via:
```bash
pip install -r deps/requirements.txt
```
The reported results were obtained with Python 3.10.8.

### Get the Datasets
The repository requires datasets to be formatted as per the CoOp / CoCoOp guidelines. 
Please follow the instructions [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to do so.


### An Effort for Reproducibility in Few-Shot Learning 
Few-Shot Learning is very sensitive to randomness in the few-shot training data, and ideally, methods
are trained on the exact same subsets to keep track of progress in the field fairly. 
To account for the above, most previous repositories would generate a training/validation split after an initial seeding and pickle it. 

However, to guarantee identical splits are generated across different repositories, not only a random seed but an identical random state at the time of sampling is required, which poses several risks and limits the usage of the `random` module throughout the repository. 

In this repo, we make two changes:
1. **Public splits.** To foster reproducibility, we openly release and distribute 165 *exact* splits that we sampled for Few-Shot Learning, corresponding to
the combinations of 11 datasets, 3 seeds, and 5 shot availabilities.
2. **Portable format.** We release our splits in a more portable `.jsonl` format and move away from the dependency on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch);

We release our training splits [here](https://drive.google.com/drive/folders/1uZN-Mjqz9U3Kjn36lj8eUhvJskBllT6j?usp=drive_link). To download and arrange them on disk automatically, run:
```bash
DATA_PATH=/your/datasets/root ./scripts/download_splits.sh
``` 

The `DATA_PATH` variable is intended to be the root of where your datasets are stored (i.e., the folder variable called $DATA in the official download instructions from the CoOp repository). You should now have `.jsonl` files in the `split_fewshot` subdirectory of each dataset, with splits for 1,2,4,8,16 shots. 


### Public results on public splits
Extended results for 2SFS-LayerNorm and CLIP-LoRA using the splits above are publicly released on Google Drive [here](https://drive.google.com/drive/folders/1ZgETDxbOqgUn-6IKIXfc0tFjXkFeyVxu?usp=sharing) (please, get in touch if you cannot access Google Drive!).  

All experiments ran with 1 NVIDIA A100 Custom 64GB on the [Leonardo Supercomputer](https://leonardo-supercomputer.cineca.eu/it/about-it/), with both algorithms running with default and fixed hyperparameters, and were averaged over random seeds 1,2, and 3.

### Few-Shot Learning!!
The entrypoint of this repository is `main.py`. Here is a snapshot of `--help`:
```
  --seed SEED
  --root_path ROOT_PATH
                        Root directory to where your datasets are stored.
  --shots SHOTS
  --dataset DATASET
  --batch_size BATCH_SIZE
  --test_batch_size TEST_BATCH_SIZE
  --workers WORKERS     num_workers for PyTorch Dataloaders
  --backbone BACKBONE
  --lr LR
  --wd WD
  --mode {cliplora,ln_only,twostage}
                        Choose which experiment to run. Choices are: 1. 'cliplora': will run CLIP-LoRA as per https://arxiv.org/abs/2405.18541; 2. 'ln_only': will do FSA by only tuning layer-
                        normalization instances according to --ln_modality and --norm_start; 3. (default) 'twostage': will run 2SFS, and you can customize it with the --peft, --n_iters and
                        --n_iters_frac arguments;
  --setting {standard,base2new}
                        Setting for the experiment. Set to 'standard' for all-to-all (train categories = eval categories) or 'base2new' otherwise.
  --debug DEBUG         Enable debugging mode (will run for a few iterations, then exit). Useful to check installation was successful.
  --results_dir RESULTS_DIR
                        Root folder to where your .csv results will be saved.
  --exp_name EXP_NAME   Experiment name (will be the basename of your .csv file).
  --n_iters N_ITERS     Shots Multiplier to get the total number of iterations. Denoted as M in the paper. Default=300.
  --n_iters_frac N_ITERS_FRAC
                        Fraction of iterations to allocate to stage 1, denoted as alpha in the paper. Default=0.6
  --peft {ln,lora,bitfit}
                        Parameter Efficient Fine-Tuning scheme to employ during the first stage when using 2SFS. Default: ln.
  --ln_modality {text,vision,both}
                        Whether to tune LayerNorm instances in only the vision, text, or both encoders. Default: 'both'.
  --ln_vision_start LN_VISION_START
                        Whether to only start tuning LayerNorm instances after a certain block of the vision encoder; Active if --ln_modality is 'both' or 'vision'. Default: 0 (tunes all
                        instances).
  --ln_text_start LN_TEXT_START
                        Same as --ln_vision_start, but for the text encoder (therefore, active if --ln_modality is 'both' or 'text'). Default: 0.
```

You can run "all-to-all" adaptation with `--setting standard` or base-to-novel generalization with `--setting base2new`. The default arguments are configured to run 2SFS (`--mode twostage`) with M=300 (`--n_iters 300`) and alpha=0.6 (`--n_iters_frac`). 
A csv file with the results will automatically be saved in `--results_dir`.

When you finish a sweep (e.g., a particular experiment for backbone x dataset x shots combo), feel free to use `summarize.py`. This will average results over different seeds and sort any `--mode` passed to `main.py` from worst to best. 

### Disclaimer & Contact
The code underwent major refactoring before the release, so feel free to file an issue or contact me at `m.farina@unitn.it` for any inquiries!

### Acknowledgements
This repo builds upon [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp/tree/main) and [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA), so huge thanks to the authors!

The authors acknowledge the CINECA award under the ISCRA initiative for the availability of high-performance computing resources and support. Matteo Farina is supported by the PRIN B-FAIR (Prot. 2022EX F3HX) project and the PAT project “AI@TN”. This work was supported by the projects EU Horizon ELIAS (No. 101120237), AI4TRUST (No.101070190), FAIR - Future AI Research (PE00000013), funded by NextGeneration EU, and carried out in the Vision and Learning joint laboratory of Fondazione Bruno Kessler and the University of Trento, Italy.

### Citation  
Please cite this work as follows if you find it useful!
```bibtex
@article{farina2025rethinking,
  title={Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages},
  author={Farina, Matteo and Mancini, Massimiliano and Iacca, Giovanni and Ricci, Elisa},
  journal={arXiv preprint arXiv:2503.11609},
  year={2025}
}
```
