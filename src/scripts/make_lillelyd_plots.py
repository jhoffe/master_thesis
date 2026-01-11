from loguru import logger

from utils.lillelyd_descriptive_plots import make_lillelyd_plots

if __name__ == "__main__":
    logger.info("Generating dataset descriptive plots for LilleLyd...")
    make_lillelyd_plots()
