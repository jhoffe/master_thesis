from utils.pairwise_comparison_utils_csr import (
    pairwise_comparison_pipeline,
)

if __name__ == "__main__":
    pairwise_comparison_pipeline(type="speaker")
    pairwise_comparison_pipeline(type="sentence")