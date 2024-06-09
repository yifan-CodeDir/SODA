
GPUID="3"

# CUDA_VISIBLE_DEVICES=$GPUID python run.py \
## accelerate training
export CUDA_VISIBLE_DEVICES=$GPUID 
python3 train.py \
    --pretrained_trans codebert \
    --experiment cross_version_codebert_token_diff \
    --train_path ../../../data/cross_version_codebert_token_diff \
    --validation_path ../../../data/cross_version_codebert_token_diff \
    --model_config ../../../src/model_configs/codebert_trans_test.json \
    --is_diff \
    2>&1 | tee log.txt