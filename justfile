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

download-coral: activate-venv
    @echo "Downloading Coral dataset..."
    uv run hf download CoRal-project/coral-v2 --repo-type dataset

download: download-coral

transfer:
    rsync -av --relative --files-from=<(git ls-files --others --cached --exclude-standard --error-unmatch 2>/dev/null | xargs -r ls -d 2>/dev/null) ./ ${HPC_HOST}:${HPC_PATH}