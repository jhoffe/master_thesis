from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def run_for_model(experiment_args: dict[str, Any], wait_for: str | None = None) -> str | None:
    job_name = f"evaluate_coral_{experiment_args['eval.model_id'].replace('/', '__')}"
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
        dependency=wait_for,
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.load_modules("cuda/12.8.0")

        command = [
            "python",
            "src/scripts/evaluate_model.py",
        ]

        command.extend(f"++{k}={v}" for k, v in experiment_args.items())

        s.command(command)

    return s.get_job_id(True)


if __name__ == "__main__":
    load_dotenv()

    experiments = [
        {
            "eval.model_id": "CoRal-project/roest-wav2vec2-315m-v2",
            "eval.no_lm": True,
        },
        {
            "eval.model_id": "CoRal-project/roest-wav2vec2-1b-v2",
            "eval.no_lm": True,
        },
        {
            "eval.model_id": "CoRal-project/roest-wav2vec2-2b-v2",
            "eval.no_lm": True,
        },
        {
            "eval.model_id": "CoRal-project/roest-whisper-large-v1",
            "eval.no_lm": False,
        },
        {
            "eval.model_id": "syvai/hviske-v2",
            "eval.no_lm": False,
        },
        {
            "eval.model_id": "syvai/hviske-v3-conversation",
            "eval.no_lm": False,
        },
        {
            "eval.model_id": "facebook/seamless-m4t-v2-large",
            "eval.no_lm": False,
            "eval.language": "dan",
        },
        {
            "eval.model_id": "openai/whisper-large-v3-turbo",
            "eval.no_lm": False,
        },
        {
            "eval.model_id": "openai/whisper-large-v3",
            "eval.no_lm": False,
        },
        {
            "eval.model_id": "facebook/wav2vec2-xls-r-2b",
            "eval.no_lm": False,
        },
    ]

    wait_for = None
    for experiment_args in experiments[3:]:
        experiment_args["eval.batch_size"] = 8

        run_for_model(
            experiment_args,
            wait_for,
        )
