set dotenv-load
set dotenv-required
set working-directory := "."
set shell := ["bash", "-uc"]


default:
    just --list

init: sync activate-venv
    @echo "Initialization complete."

activate-venv:
    @echo "Activating virtual environment..."
    source .venv/bin/activate

sync:
    uv sync

transfer:
    rsync -av --relative --files-from=<(git ls-files --others --cached --exclude-standard --error-unmatch 2>/dev/null | xargs -r ls -d 2>/dev/null) ./ ${HPC_HOST}:${HPC_PATH}

download-results:
    rsync -av ${HPC_HOST}:${HPC_PATH}/experiments .

submit-eval: activate-venv transfer
    python jobs/submit_evaluation.py

nemo-download-coral: activate-venv
    @echo "Converting Coral Training Set"
    python external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_CORAL_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="train" \
        use_auth_token=True \
        streaming=True

    @echo "Converting Coral Validation Set"
    python external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_CORAL_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="val" \
        use_auth_token=True \
        streaming=True

    @echo "Converting Coral Test Set"
    python external/convert_hf_dataset_to_nemo.py \
        output_dir=${NEMO_CORAL_DATASET_PATH} \
        path="CoRal-project/coral-v2" \
        name="read_aloud" \
        split="test" \
        use_auth_token=True \
        streaming=True

nemo-convert-coral: activate-venv
    @echo "Converting Coral Training Set"

    @echo "Creating tarred dataset for Coral Training Set"
    python external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_CORAL_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_CORAL_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/readtrain \
        --num_shards=296 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Coral Validation Set"
    python external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_CORAL_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/val/val_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_CORAL_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/val \
        --num_shards=4 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

    @echo "Creating tarred dataset for Coral Test Set"
    python external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_CORAL_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/test/test_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_CORAL_DATASET_PROCESSED_PATH}/CoRal-project_coral-v2_read-aloud/test \
        --num_shards=12 \
        --max_duration=120 \
        --min_duration=0.01 \
        --shuffle --shuffle_seed=1 \
        --sort_in_shards \
        --force_codec=flac \
        --workers=-1

remove-intermediate:
    rm -rf ${NEMO_CORAL_DATASET_PATH}/CoRal-project

nemo-dataset-prepare: nemo-download-coral nemo-convert-coral remove-intermediate
    @echo "Nemo dataset preparation complete."