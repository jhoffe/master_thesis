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
        streaming=True

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
        streaming=True

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

    @echo "Creating tarred dataset for Fleurs Training Set"
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

    @echo "Creating tarred dataset for Fleurs Validation Set"
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

    @echo "Creating tarred dataset for Fleurs Test Set"
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

# Converts Coral dataset to tarred audio dataset format
[group('nemo')]
nemo-convert-coral:
    @echo "Converting Coral Training Set"

    @echo "Creating tarred dataset for Coral Training Set"
    {{python}} external/convert_to_tarred_audio_dataset.py \
        --manifest_path ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --target_dir=${NEMO_DATASET_PATH}/CoRal-project_coral-v2_read-aloud/train \
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
        --target_dir=${NEMO_DATASET_PATH}/CoRal-project_coral-v2_read-aloud/val \
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
        --target_dir=${NEMO_DATASET_PATH}/CoRal-project_coral-v2_read-aloud/test \
        --num_shards=12 \
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

# Estimates duration bins for specified model and dataset
[group('nemo')]
nemo-estimate-duration-bins-2d:
    @echo "Estimating duration bins for Coral Training Set for nvidia/canary-1b-v2"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --buckets 30 \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/canary-1b-v2"

    @echo "Estimating duration bins for Coral Training Set for nvidia/parakeet-tdt-0.6b-v3"
    {{python}} external/estimate_duration_bins_2d.py \
        ${NEMO_DATASET_PATH}/CoRal-project/coral-v2/read_aloud/train/train_CoRal-project_coral-v2_manifest.json \
        --buckets 30 \
        --sub-buckets 5 \
        --tokenizer-from-pretrained-model="nvidia/parakeet-tdt-0.6b-v3"

# Runs OOMptimizer for specified model and buckets to generate optimal batch sizes
[group('nemo')]
nemo-oomptimize:
    @echo "OOMptimizing nvidia/canary-1b-v2"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/canary-1b-v2" \
        --buckets="[[3.360,12],[3.360,13],[3.360,15],[3.360,17],[3.360,27],[3.780,13],[3.780,15],[3.780,17],[3.780,19],[3.780,30],[4.080,14],[4.080,16],[4.080,18],[4.080,21],[4.080,31],[4.320,16],[4.320,18],[4.320,20],[4.320,22],[4.320,33],[4.560,16],[4.560,19],[4.560,21],[4.560,23],[4.560,35],[4.800,17],[4.800,19],[4.800,21],[4.800,24],[4.800,36],[5.040,18],[5.040,20],[5.040,23],[5.040,25],[5.040,39],[5.220,18],[5.220,20],[5.220,23],[5.220,26],[5.220,38],[5.400,19],[5.400,22],[5.400,24],[5.400,26],[5.400,38],[5.640,19],[5.640,22],[5.640,25],[5.640,28],[5.640,43],[5.820,20],[5.820,22],[5.820,25],[5.820,28],[5.820,42],[6.000,21],[6.000,24],[6.000,26],[6.000,29],[6.000,43],[6.180,21],[6.180,25],[6.180,27],[6.180,30],[6.180,43],[6.420,21],[6.420,24],[6.420,27],[6.420,30],[6.420,45],[6.600,22],[6.600,25],[6.600,27],[6.600,31],[6.600,45],[6.820,23],[6.820,27],[6.820,30],[6.820,33],[6.820,47],[7.020,23],[7.020,26],[7.020,29],[7.020,32],[7.020,47],[7.260,24],[7.260,27],[7.260,30],[7.260,34],[7.260,49],[7.500,25],[7.500,28],[7.500,31],[7.500,34],[7.500,50],[7.740,25],[7.740,28],[7.740,31],[7.740,35],[7.740,52],[7.980,26],[7.980,29],[7.980,32],[7.980,36],[7.980,52],[8.280,26],[8.280,30],[8.280,33],[8.280,37],[8.280,55],[8.580,27],[8.580,31],[8.580,34],[8.580,37],[8.580,53],[8.940,28],[8.940,32],[8.940,35],[8.940,39],[8.940,56],[9.300,29],[9.300,33],[9.300,36],[9.300,39],[9.300,56],[9.780,29],[9.780,33],[9.780,37],[9.780,40],[9.780,58],[10.330,31],[10.330,35],[10.330,38],[10.330,42],[10.330,62],[11.160,31],[11.160,35],[11.160,39],[11.160,43],[11.160,63],[12.480,33],[12.480,37],[12.480,40],[12.480,44],[12.480,67],[37.920,34],[37.920,39],[37.920,43],[37.920,47],[37.920,74]]"

    @echo "OOMptimizing nvidia/parakeet-tdt-0.6b-v3"
    {{python}} external/oomptimizer.py \
        --pretrained-name="nvidia/parakeet-tdt-0.6b-v3" \
        --buckets="[[3.360,13],[3.360,15],[3.360,16],[3.360,18],[3.360,30],[3.780,15],[3.780,17],[3.780,19],[3.780,21],[3.780,33],[4.080,16],[4.080,18],[4.080,20],[4.080,22],[4.080,34],[4.320,17],[4.320,20],[4.320,22],[4.320,24],[4.320,36],[4.560,18],[4.560,20],[4.560,23],[4.560,26],[4.560,38],[4.800,18],[4.800,21],[4.800,23],[4.800,26],[4.800,40],[5.040,19],[5.040,22],[5.040,25],[5.040,28],[5.040,42],[5.220,20],[5.220,22],[5.220,25],[5.220,28],[5.220,42],[5.400,21],[5.400,24],[5.400,26],[5.400,29],[5.400,44],[5.640,21],[5.640,24],[5.640,27],[5.640,31],[5.640,46],[5.820,22],[5.820,25],[5.820,27],[5.820,31],[5.820,46],[6.000,23],[6.000,26],[6.000,28],[6.000,31],[6.000,47],[6.180,24],[6.180,27],[6.180,30],[6.180,33],[6.180,49],[6.420,23],[6.420,27],[6.420,30],[6.420,33],[6.420,49],[6.600,25],[6.600,28],[6.600,30],[6.600,34],[6.600,50],[6.820,26],[6.820,29],[6.820,32],[6.820,36],[6.820,52],[7.020,26],[7.020,29],[7.020,32],[7.020,35],[7.020,52],[7.260,26],[7.260,30],[7.260,34],[7.260,37],[7.260,54],[7.500,27],[7.500,31],[7.500,34],[7.500,38],[7.500,54],[7.740,28],[7.740,31],[7.740,34],[7.740,38],[7.740,57],[7.980,29],[7.980,32],[7.980,35],[7.980,39],[7.980,57],[8.280,29],[8.280,33],[8.280,37],[8.280,40],[8.280,59],[8.580,30],[8.580,34],[8.580,37],[8.580,41],[8.580,59],[8.940,30],[8.940,35],[8.940,38],[8.940,42],[8.940,60],[9.300,32],[9.300,36],[9.300,39],[9.300,43],[9.300,60],[9.780,32],[9.780,37],[9.780,40],[9.780,44],[9.780,63],[10.330,34],[10.330,38],[10.330,42],[10.330,46],[10.330,67],[11.160,34],[11.160,39],[11.160,42],[11.160,47],[11.160,67],[12.480,36],[12.480,40],[12.480,44],[12.480,48],[12.480,74],[37.920,38],[37.920,43],[37.920,47],[37.920,52],[37.920,81]]"

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