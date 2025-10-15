from __future__ import annotations

import os
from time import sleep
from typing import Any

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def run_for_model(
    dataset_arg: str,
    model_arg: str,
    experiment_args: dict[str, Any] | None = None,
) -> str | None:
    job_name = f"eval-{model_arg}-{dataset_arg}"
    if dataset_arg == "coral":
        walltime = "01:30"
        if model_arg in ["hviske-v2", "hviske-v3-conversation", "roest-whisper-large-v1", "whisper-large-v3"]:
            walltime = "03:00"
    else:
        walltime = "00:20"
        if model_arg in ["hviske-v2", "hviske-v3-conversation", "roest-whisper-large-v1", "whisper-large-v3"]:
            walltime = "00:40"

    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        num_cores=8,
        walltime=walltime,
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
            "model=" + model_arg,
            "dataset=" + dataset_arg,
        ]

        if experiment_args is not None:
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
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "roest-wav2vec2-315m-v2",
        "roest-wav2vec2-1B-v2",
        "roest-wav2vec2-2B-v2",
    ]

    for experiment in models:
        run_for_model(
            dataset_arg="coral",
            model_arg=experiment,
        )

        run_for_model(
            dataset_arg="fleurs",
            model_arg=experiment,
        )

        sleep(5)
