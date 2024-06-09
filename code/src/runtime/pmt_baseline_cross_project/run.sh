
GPUID="1"

# CUDA_VISIBLE_DEVICES=$GPUID python run.py \
## accelerate training
export CUDA_VISIBLE_DEVICES=$GPUID 
python3 train.py \
    --experiment seshat_cross_project_pmt_baseline_fold1 \
    --train_path ../../../data/seshat_cross_project_pmt_baseline_fold1 \
    --validation_path ../../../data/seshat_cross_project_pmt_baseline_fold1 \
    --model_config ../../../src/model_configs/pmt_baseline.json\
    --sent_vocab_path ../../../data/seshat_cross_project_pmt_baseline_fold1 \
    --body_vocab_path ../../../data/seshat_cross_project_pmt_baseline_fold1 \
    2>&1 | tee log.txt