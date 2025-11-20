# master_thesis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Environment Variables

Copy `.env.default` to `.env` and fill in the values you need:

| Variable                                           | Required?                   | Purpose / notes                                                                   | Example                      |
| -------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------- | ---------------------------- |
| `HF_AUTH_TOKEN`                                    | Yes for private HF datasets | Token used when downloading Hugging Face datasets (e.g., CoRal/Fleurs with auth). | `hf_xxx`                     |
| `HF_HUB_CACHE`                                     | Optional                    | Local Hugging Face cache path.                                                    | `./data/raw/huggingface`     |
| `NEMO_DATASET_PATH`                                | Yes                         | Where raw HF datasets are written by the conversion scripts.                      | `data/raw`                   |
| `NEMO_DATASET_PROCESSED_PATH`                      | Yes                         | Where processed/tarred NeMo data is stored and read by configs.                   | `data/processed`             |
| `WANDB_API_KEY`                                    | Yes if logging              | Auth key for Weights & Biases.                                                    | `xxxxxxxx`                   |
| `WANDB_PROJECT`                                    | Yes if logging              | Target W&B project (also used by NeMo exp_manager).                               | `your-project`               |
| `WANDB_ENTITY`                                     | Yes if logging              | W&B entity/org.                                                                   | `your-entity`                |
| `HPC_HOST`                                         | Yes for rsync helpers       | SSH host for syncing results via `just` recipes.                                  | `user@cluster`               |
| `HPC_PATH`                                         | Yes for job submission      | Remote working directory used by LSF submissions.                                 | `/cluster/home/user/project` |
| `HPC_EMAIL_ADDRESS`                                | Optional                    | Email for job notifications.                                                      | `you@example.com`            |
| `HPC_NOTIFY_ON_START` / `HPC_NOTIFY_ON_COMPLETION` | Optional                    | Set to `1` to enable LSF email notifications.                                     | `0`                          |
| `N_JOBS`                                           | Optional                    | Parallel workers for test-set prep (defaults to CPU count).                       | `8`                          |
| `RANDOM_SEED`                                      | Optional                    | Global seed used in scripts/configs.                                              | `42`                         |

## Finetuning Experiments

| Experiment                                        | Base model                    | Config path                                                        | Data / augmentations                                                                                                                              |
| ------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `canary-finetune`                                 | `nvidia/canary-1b-v2`         | `nemo_config/canary-finetune.yaml`                                 | CoRal read-aloud train set (`train_ds_canary_30_buckets`), 30 lhotse buckets, no spec augment, base preprocessor.                                 |
| `canary-finetune_spec-aug`                        | `nvidia/canary-1b-v2`         | `nemo_config/canary-finetune_spec-aug.yaml`                        | Same dataset/bucketing as above with spec augment (2 freq masks, 2 time masks).                                                                   |
| `canary-finetune_speech-perturbations`            | `nvidia/canary-1b-v2`         | `nemo_config/canary-finetune_speech-perturbations.yaml`            | Speech-perturbed CoRal variant (`train_ds_canary_30_buckets_speech_perturbations`), 30 buckets, no spec augment.                                  |
| `canary-finetune_spec-aug_speech-perturbations`   | `nvidia/canary-1b-v2`         | `nemo_config/canary-finetune_spec-aug_speech-perturbations.yaml`   | Speech-perturbed train set with 30 buckets + spec augment enabled.                                                                                |
| `canary-finetune_pitch-shift`                     | `nvidia/canary-1b-v2`         | `nemo_config/canary-finetune_pitch-shift.yaml`                     | CoRal 30-bucket train set, pitch-shift preprocessor (±2 semitone steps at 0.5 prob), spec augment disabled.                                       |
| `parakeet-finetune`                               | `nvidia/parakeet-tdt-0.6b-v3` | `nemo_config/parakeet-finetune.yaml`                               | CoRal read-aloud tarred train set (`train_ds_parakeet`, batch size 16), no spec augment.                                                          |
| `parakeet-finetune_spec-aug`                      | `nvidia/parakeet-tdt-0.6b-v3` | `nemo_config/parakeet-finetune_spec-aug.yaml`                      | Same tarred train set with spec augment enabled.                                                                                                  |
| `parakeet-finetune_speech-perturbations`          | `nvidia/parakeet-tdt-0.6b-v3` | `nemo_config/parakeet-finetune_speech-perturbations.yaml`          | Speech perturbations block configured (`train_ds_parakeet_speech_perturbations` available), spec augment disabled; validation/test use base sets. |
| `parakeet-finetune_spec-aug_speech-perturbations` | `nvidia/parakeet-tdt-0.6b-v3` | `nemo_config/parakeet-finetune_spec-aug_speech-perturbations.yaml` | Speech perturbations block + spec augment enabled; uses base validation/test datasets.                                                            |
| `parakeet-finetune_pitch-shift`                   | `nvidia/parakeet-tdt-0.6b-v3` | `nemo_config/parakeet-finetune_pitch-shift.yaml`                   | CoRal tarred train set, pitch-shift preprocessor (±2 semitone steps at 0.5 prob), spec augment disabled.                                          |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         master_thesis and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── master_thesis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes master_thesis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
