# FCN Project New

This folder contains the final runnable code for the FCN pipeline.

## Main Files

- `preprocess/finetune_m3e_mnrl.py`: fine-tunes the Stage 1 encoder on the labeled data.
- `preprocess/build_reason_proto_targeted.py`: builds the targeted fine-tuning dataset used by the encoder script.
- `stage1/FCN_stage1_Candidate.py`: runs Stage 1 candidate retention and writes the frozen Stage 1 outputs.
- `stage2/FCN_stage2_structured.py`: runs the formal Stage 2 model with one-dimensional Stage 2A evidence and exact Stage 2B recovery.
- `stage2/FCN_stage2_structure_ablation.py`: compares M0 / M1 / M2 structure variants.
- `stage2/FCN_stage2A_ablation.py`: compares the Stage 2A feature settings.
- `stage2/FCN_main_experiments_3split.py`: runs the main baselines, GraphSAGE, bootstrap, and final FCN comparison.

## Recommended Order

1. Fine-tune the encoder.
2. Run Stage 1.
3. Run the formal Stage 2 script.
4. Run the Stage 2 ablations if needed.
5. Run the main experiment script.

## Key Data And Outputs

- `train data/fcn_30firms_full_labeled.csv`: final labeled dataset.
- `train data/targeted_reason_proto_augmented.csv`: targeted fine-tuning data.
- `checkpoints/fcn-m3e-base-mnrl/`: fine-tuned encoder checkpoint.

The scripts are intended to run directly with their default paths.
