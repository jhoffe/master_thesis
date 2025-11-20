from loguru import logger

from utils.evaluation_plots import (
    make_plots,
)


def main():
    logger.info("Starting evaluation plots generation...")
    make_plots()
    logger.info("Evaluation plots generation completed.")


if __name__ == "__main__":
    main()
