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
    rsync -av ${HPC_HOST}:${HPC_PATH}/carbon_logs .

submit-eval: activate-venv transfer
    python jobs/submit_evaluation.py