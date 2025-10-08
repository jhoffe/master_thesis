from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def run_for_model(model_id: str, wav2vec: bool) -> None:
    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=f"evaluate_coral_{model_id.replace('/', '__')}",
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        num_cores=8,
        walltime="02:00",
        memory="4GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file="logs/evaluate_coral.%J.out",
        error_file="logs/evaluate_coral.%J.err",
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.load_modules("cuda/12.8.0")

        command = [
            "python",
            "src/scripts/evaluate_model.py",
            f"++eval.model_id={model_id}",
            "++eval.batch_size=8",
        ]

        if not wav2vec:
            command.append("++eval.no_lm=true")
        else:
            command.append("++eval.no_lm=false")

        s.command(command)


if __name__ == "__main__":
    load_dotenv()

    models = [
        ("CoRal-project/roest-wav2vec2-315m-v2", True),
        ("CoRal-project/roest-wav2vec2-1b-v2", True),
        ("CoRal-project/roest-wav2vec2-2b-v2", True),
        ("CoRal-project/roest-whisper-large-v1", False),
        ("syvai/hviske-v2", False),
        ("syvai/hviske-v3-conversation", False),
        ("facebook/seamless-m4t-v2-large", False),
        ("openai/whisper-large-v3-turbo", False),
        ("openai/whisper-large-v3", False),
        ("facebook/wav2vec2-xls-r-2b", False),
    ]

    for model, wav2vec in models:
        run_for_model(model, wav2vec)
