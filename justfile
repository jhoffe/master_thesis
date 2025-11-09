set dotenv-load := true
set dotenv-required := true
set working-directory := "."
set shell := ["bash", "-uc"]

python := "./.venv/bin/python"

default:
    just --list

init:
    @echo "Setting up virtual environment..."
    uv sync
    @echo "Initialization complete."

# Transfers all tracked and untracked files to HPC
[group('hpc')]
transfer:
    rsync -av --relative --files-from=<(git ls-files --others --cached --exclude-standard --error-unmatch 2>/dev/null | xargs -r ls -d 2>/dev/null) ./ ${HPC_HOST}:${HPC_PATH}

# Downloads results from HPC to local machine
[group('hpc')]
download-results:
    rsync -av ${HPC_HOST}:${HPC_PATH}/experiments .
    rsync -av ${HPC_HOST}:${HPC_PATH}/carbon_logs .

# Downloads specs from HPC to local machine
[group('hpc')]
download-specs:
    rsync -av ${HPC_HOST}:${HPC_PATH}/specs specs/

download-jonas-results:
    rsync -av ${HPC_HOST}:${HPC_PATH_JONAS}/experiments/evaluate_model/canary-1b-v2_finetuned_spec-aug_coral-v2_read_aloud_test .
    rsync -av ${HPC_HOST}:${HPC_PATH_JONAS}/experiments/evaluate_model/canary-1b-v2_finetuned_spec-aug_fleurs_da_dk_test .
    rsync -av ${HPC_HOST}:${HPC_PATH_JONAS}/experiments/evaluate_model/parakeet-tdt-0.6b-v3_finetuned_spec-aug_coral-v2_read_aloud_test .
    rsync -av ${HPC_HOST}:${HPC_PATH_JONAS}/experiments/evaluate_model/parakeet-tdt-0.6b-v3_finetuned_spec-aug_fleurs_da_dk_test .
    #rsync -av ${HPC_HOST}:${HPC_PATH_JONAS}/carbon_logs .

# Downloads a specific path from HPC to local machine
[group('hpc')]
download path:
    rsync -av ${HPC_HOST}:${HPC_PATH}/{{path}} {{path}}

# Downloads and converts Coral dataset to NeMo format
[group('nemo')]
nemo-download-coral:
    @echo "Converting Coral Training Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="train" \
        text_column="text" \
        pnc=true \
        use_auth_token=True \
        streaming=True \
        speed_perturb="[0.9,1.1]"

    @echo "Converting Coral Validation Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="val" \
        text_column="text" \
        pnc=false \
        use_auth_token=True \
        streaming=True

    @echo "Converting Coral Test Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="test" \
        text_column="text" \
        pnc=false \
        use_auth_token=True \
        streaming=True

# Downloads and converts Fleurs dataset to NeMo format
[group('nemo')]
nemo-download-fleurs:
    @echo "Converting Fleurs Training Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="google/fleurs" \
        name="da_dk" \
        split="train" \
        text_column="raw_transcription" \
        pnc=true \
        use_auth_token=True \
        streaming=True \
        speed_perturb="[0.9,1.1]"

    @echo "Converting Fleurs dev Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="google/fleurs" \
        name="da_dk" \
        split="validation" \
        text_column="raw_transcription" \
        pnc=false \
        use_auth_token=True \
        streaming=True

    @echo "Converting Fleurs Test Set"
    {{python}} external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_DATASET_PATH} \
        path="google/fleurs" \
        name="da_dk" \
        split="test" \
        text_column="raw_transcription" \
        pnc=false \
        use_auth_token=True \
        streaming=True

# Converts Fleurs dataset to tarred audio dataset format
[group('nemo')]
nemo-convert-fleurs:
    @echo "Converting Fleurs Training Set"

    @echo "Creating tarred dataset for Fleurs Training Set without speed perturbation"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/google/fleurs/da_dk/train/train_google_fleurs_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/google_fleurs/train \
        --num_shards=5 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Fleurs Validation Set without speed perturbation"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/google/fleurs/da_dk/validation/validation_google_fleurs_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/google_fleurs/validation \
        --num_shards=2 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Fleurs Test Set without speed perturbation"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/google/fleurs/da_dk/test/test_google_fleurs_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/google_fleurs/test \
        --num_shards=2 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Fleurs Training Set with speed perturbation"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/google/fleurs/da_dk/train/train_google_fleurs_manifest_sp.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/google_fleurs_sp/train \
        --num_shards=15 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

# Converts Coral dataset to tarred audio dataset format
[group('nemo')]
nemo-convert-coral:
    @echo "Converting Coral Training Set"

    @echo "Creating tarred dataset for Coral Training Set"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/train \
        --num_shards=296 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Coral Validation Set"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/val/val_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/val \
        --num_shards=4 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Coral Test Set"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/test/test_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/test \
        --num_shards=12 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Coral Training Set"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest_sp.json \
        --target_dir=${NEMO_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud_sp/train \
        --num_shards=882 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

# Prepares Coral dataset by downloading and converting
[group('nemo')]
nemo-coral-prepare: nemo-download-coral nemo-convert-coral
    @echo "Coral dataset preparation complete."

# Prepares Fleurs dataset by downloading and converting
[group('nemo')]
nemo-fleurs-prepare: nemo-download-fleurs nemo-convert-fleurs
    @echo "Fleurs dataset preparation complete."

# Prepares both Coral and Fleurs datasets
[group('nemo')]
nemo-dataset-prepare: nemo-coral-prepare nemo-fleurs-prepare 
    @echo "Nemo dataset preparation complete."

[group('nemo')]
nemo-estimate-duration-bins-2d-canary buckets='12':
    @echo "Estimating duration bins for Coral Training Set for nvidia/canary-1b-v2"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --buckets {{buckets}} \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/canary-1b-v2"

[group('nemo')]
nemo-estimate-duration-bins-2d-canary-sp buckets='12':
    @echo "Estimating duration bins for Coral Training Set for nvidia/canary-1b-v2 with speed perturbation"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest_sp.json \
        --buckets {{buckets}} \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/canary-1b-v2"

[group('nemo')]
nemo-estimate-duration-bins-2d-parakeet buckets='12':
    @echo "Estimating duration bins for Coral Training Set for nvidia/parakeet-tdt-0.6b-v3"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --buckets {{buckets}} \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/parakeet-tdt-0.6b-v3"

[group('nemo')]
nemo-estimate-duration-bins-2d-parakeet-sp buckets='12':
    @echo "Estimating duration bins for Coral Training Set for nvidia/parakeet-tdt-0.6b-v3 with speed perturbation"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest_sp.json \
        --buckets {{buckets}} \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/canary-1b-v2"

# Estimates duration bins for specified model and dataset
[group('nemo')]
nemo-estimate-duration-bins-2d: nemo-estimate-duration-bins-2d-canary nemo-estimate-duration-bins-2d-canary-sp nemo-estimate-duration-bins-2d-parakeet nemo-estimate-duration-bins-2d-parakeet-sp
    @echo "Duration bin estimation complete."

# Runs OOMptimizer for specified model and buckets to generate optimal batch sizes
[group('nemo')]
nemo-oomptimize:
    @echo "OOMptimizing nvidia/canary-1b-v2 with 30 buckets"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/canary-1b-v2" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_canary_30_buckets.yaml" \
        --memory-fraction=0.9
    
    @echo "OOMptimizing nvidia/canary-1b-v2 with 30 buckets with speed perturbations"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/canary-1b-v2" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_canary_30_buckets_speech_perturbations.yaml" \
        --memory-fraction=0.9

    @echo "OOMptimizing nvidia/canary-1b-v2 with 12 buckets"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/canary-1b-v2" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_canary_12_buckets.yaml" \
        --memory-fraction=0.9
    
    @echo "OOMptimizing nvidia/canary-1b-v2 with 12 buckets with speed perturbations"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/canary-1b-v2" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_canary_12_buckets_speech_perturbations.yaml" \
        --memory-fraction=0.9

    @echo "OOMptimizing nvidia/parakeet-tdt-0.6b-v3 with 30 buckets"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/parakeet-tdt-0.6b-v3" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_parakeet_30_buckets.yaml" \
        --memory-fraction=0.9

    @echo "OOMptimizing nvidia/parakeet-tdt-0.6b-v3 with 30 buckets with speech perturbations"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/parakeet-tdt-0.6b-v3" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_parakeet_30_buckets_speech_perturbations.yaml" \
        --memory-fraction=0.9

    @echo "OOMptimizing nvidia/parakeet-tdt-0.6b-v3 with 12 buckets"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/parakeet-tdt-0.6b-v3" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_parakeet_12_buckets.yaml" \
        --memory-fraction=0.9

    @echo "OOMptimizing nvidia/parakeet-tdt-0.6b-v3 with 12 buckets with speech perturbations"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/parakeet-tdt-0.6b-v3" \
        --dataset-config-path="nemo_config/model/train_ds/train_ds_parakeet_12_buckets_speech_perturbations.yaml" \
        --memory-fraction=0.9

# Submit OOMptimizer job
[group('jobs')]
submit-oomptimize gpumem='80gb': transfer
    @echo "Submitting OOMptimizer job"
    {{python}} jobs/submit_oomptimizer.py --gpu-mem={{gpumem}}
    @echo "Finished submitting OOMptimizer job"

[group('jobs')]
submit-train-kenlm: transfer
    @echo "Submitting KenLM training job"
    {{python}} jobs/submit_train_kenlm.py
    @echo "Finished submitting KenLM training job"


# Submit all finetuning jobs
[group('jobs')]
submit-finetune: transfer
    {{python}} jobs/submit_model_finetuning.py

# Submit all evaluation jobs
[group('jobs')]
submit-eval: transfer
    {{python}} jobs/submit_evaluation.py

# List all submitted jobs
[group('jobs')]
jobs:
    uv run subjob jobs list -e

# Watch the jobs every 15 seconds
[group('jobs')]
wjobs:
    watch -n 15 uv run subjob jobs list -e

[group('lm')]
train-kenlm-parakeet:
    {{python}} external/train_kenlm.py \
        nemo_model_file="nvidia/parakeet-tdt-0.6b-v3" \
        train_paths="[${NEMO_DATASET_PROCESSED_PATH}/lm/lm_training.jsonl]" \
        kenlm_model_file="kenlm_parakeet_0.6b.arpa" \
        ngram_length=6 \
        save_nemo=true \
        verbose=1


[group('lm')]
train-kenlm: train-kenlm-parakeet
    @echo "KenLM training complete."