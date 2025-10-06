from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def run_for_model(model_conf: str) -> None:
    opts = LSFSubmissionOptions(
        queue="gpuv100",
        job_name=f"evaluate_coral_{model_conf}",
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        num_cores=8,
        walltime="01:00",
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

        s.command(
            [
                "python",
                "src/scripts/evaluate_model.py",
                f"eval={model_conf}",
                "++eval.batch_size=32",
            ]
        )


if __name__ == "__main__":
    load_dotenv()
    run_for_model("wav2vec2-roest-315m-v2")
