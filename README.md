# FCN Experiment Scripts

This package contains the Python scripts used for the FCN follow-up network experiments.

## Directory Structure

- `preprocess/`: LLM-assisted labeling demo and encoder fine-tuning.
- `stage1/`: candidate-edge retention and Stage 1 scoring.
- `stage2/`: structured graph recovery and baseline comparisons.
- `experiments/`: ablation, robustness, diagnostic, and plotting scripts.

## Typical Workflow

1. Prepare the labeled dataset and session split manifest locally.
2. Optionally fine-tune the retrieval encoder with `preprocess/finetune_m3e_mnrl.py`.
3. Run Stage 1 with `stage1/FCN_stage1_Candidate.py`.
4. Run Stage 2 with `stage2/FCN_stage2_structured.py`.
5. Run baseline comparisons with `stage2/FCN_main_experiments_3split.py`.
6. Run supplementary analyses under `experiments/`.

## Dependencies

The scripts mainly use:

- Python 3.9+
- pandas, numpy, scikit-learn
- torch
- sentence-transformers
- transformers
- modelscope
- openai
- matplotlib

Install the exact versions according to your local environment and GPU/CUDA setup.