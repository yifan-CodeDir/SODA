
GPUID="2"

# CUDA_VISIBLE_DEVICES=$GPUID python run.py \
## accelerate training
export CUDA_VISIBLE_DEVICES=$GPUID 
python3 train.py \
    --pretrained_trans codebert \
    --experiment cross_project_codebert_token_diff_fold5 \
    --train_path ../../../data/cross_project_codebert_token_diff_fold5 \
    --validation_path ../../../data/cross_project_codebert_token_diff_fold5 \
    --model_config ../../../src/model_configs/codebert_trans_test.json \
    --is_diff \
    2>&1 | tee log.txt