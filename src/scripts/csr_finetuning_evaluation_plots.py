from loguru import logger

from utils.csr_finetuning_evaluation_plots import (
    make_plots,
)


def main():
    logger.info("Starting speaker evaluation plots generation...")
    make_plots(type="speaker")
    logger.info("Speaker evaluation plots generation completed.")

    logger.info("Starting sentence evaluation plots generation...")
    make_plots(type="sentence")
    logger.info("Sentence evaluation plots generation completed.")


if __name__ == "__main__":
    main()
