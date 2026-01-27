import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def submit_job(config_name: str, walltime: str, sentence_cv: bool = False, parakeet: bool = True):
    job_name = (
        f"finetune-model-cv-{config_name}-sentence"
        if sentence_cv
        else f"finetune-model-cv-{config_name}"
    )

    opts = LSFSubmissionOptions(
        queue="gpuh100",
        job_name=job_name,
        num_cores=16,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime=walltime,
        memory="4GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        environment={
            "NUMBA_CUDA_USE_NVIDIA_BINDING": "0",
            "WANDB_JOB_TYPE": "finetune_csr",
        },
        email=os.environ.get("HPC_EMAIL_ADDRESS"),
        notify_on_start=os.environ.get("HPC_NOTIFY_ON_START") == "1",
        notify_on_completion=os.environ.get("HPC_NOTIFY_ON_COMPLETION") == "1",
    )

    data_path = (
        "../datasets/processed/LilleLyd/cv_folds_sentence"
        if sentence_cv
        else "../datasets/processed/LilleLyd/cv_folds"
    )
    train_manifest_paths = [
        os.path.join(data_path, f"fold_{i}/train_manifest.jsonl")
        for i in range(1, 5 if not sentence_cv else 6)
    ]
    test_manifest_paths = [
        os.path.join(data_path, f"fold_{i}/test_manifest.jsonl")
        for i in range(1, 5 if not sentence_cv else 6)
    ]

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        for i, (train_manifest_path, test_manifest_path) in enumerate(
            zip(train_manifest_paths, test_manifest_paths)
        ):
            name = (
                f"{config_name}_sentence_cv-{i + 1}"
                if sentence_cv
                else f"{config_name}_cv-{i + 1}"
            )

            name = (
                name.replace("spec-aug", "SA")
                .replace("speed-perturbations", "SP")
                .replace("pitch-shift", "PS")
            )

            s.command(
                [
                    "python",
                    "src/scripts/train_model.py",
                    "--config-name",
                    f"{config_name}.yaml",
                    f"++model.train_ds.manifest_filepath='{train_manifest_path}'",
                    f"++name={name}",
                    f"++exp_manager.checkpoint_callback_params.filename={name}",
                    "~model.optim.sched",
                ]
            )

            model_eval_name = (
                "parakeet-tdt-0.6b-v3_finetune" if parakeet else "canary-1b-v2_finetune"
            )

            s.command(
                [
                    "python",
                    "src/scripts/evaluate_model.py",
                    f"model={model_eval_name}",
                    f"++model.name={name}",
                    f"++model.restore_from=$(cat {name}.artifact-reference)/{name}.nemo",
                    "dataset=lillelyd_full",
                    f"++dataset.manifest_path={test_manifest_path}",
                ]
            )

            s.command(
                [
                    "python",
                    "src/scripts/evaluate_model.py",
                    f"model={model_eval_name}",
                    f"++model.restore_from=$(cat {name}.artifact-reference)/{name}.nemo",
                    f"++model.name={name}",
                    "dataset=coral",
                ]
            )

            s.command(
                [
                    "python",
                    "src/scripts/evaluate_model.py",
                    f"model={model_eval_name}",
                    f"++model.name={name}",
                    f"++model.restore_from=$(cat {name}.artifact-reference)/{name}.nemo",
                    "dataset=fleurs",
                ]
            )


def main():
    """
    Submit model finetuning jobs to the cluster.
    """
    load_dotenv()

    submit_job("parakeet-finetune_spec-aug_ll", "16:00")
    submit_job("parakeet-finetune_spec-aug_pitch-shift_ll", "16:00")
    submit_job("canary-finetune_spec-aug_speed-perturbations_ll", "16:00", parakeet=False)

    submit_job("parakeet-finetune_spec-aug_ll", "16:00", sentence_cv=True)
    submit_job("parakeet-finetune_spec-aug_pitch-shift_ll", "16:00", sentence_cv=True)
    submit_job(
        "canary-finetune_spec-aug_speed-perturbations_ll",
        "16:00",
        sentence_cv=True,
        parakeet=False,
    )


if __name__ == "__main__":
    main()
