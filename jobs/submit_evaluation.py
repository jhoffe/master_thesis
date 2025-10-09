from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode
from utils import data


def run_for_model(dataset_arg: str, eval_arg: str, experiment_args: dict[str, Any]) -> str | None:
    job_name = f"eval-{eval_arg}-{dataset_arg}"
    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        num_cores=8,
        walltime="02:00",
        memory="4GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.load_modules("cuda/12.8.0")

        command = [
            "python",
            "src/scripts/evaluate_model.py",
            "eval=" + eval_arg,
            "dataset=" + dataset_arg,
        ]

        command.extend(f"++{k}={v}" for k, v in experiment_args.items())

        s.command(command)

    return s.get_job_id(True)


if __name__ == "__main__":
    load_dotenv()

    models = [
        "hviske-v2",
        "hviske-v3-conversation",
        "roest-whisper-large-v1",
        "seamless-m4t-v2-large",
        "wav2vec2-xls-r-2b",
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "roest-wav2vec2-315m-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-2B-v2",
    ]

    for experiment in models:
        experiment_args = {
            "eval.batch_size": 8,
        }

        run_for_model(
            dataset_arg="coral",
            eval_arg=experiment,
            experiment_args=experiment_args,
        )

        experiment_args["eval.text_column"] = "transcription"

        run_for_model(
            dataset_arg="fleurs",
            eval_arg=experiment,
            experiment_args=experiment_args,
        )
