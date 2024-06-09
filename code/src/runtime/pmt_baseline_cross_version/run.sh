
GPUID="2"

# CUDA_VISIBLE_DEVICES=$GPUID python run.py \
## accelerate training
export CUDA_VISIBLE_DEVICES=$GPUID 
python3 train.py \
    --experiment seshat_cross_version_pmt_baseline \
    --train_path ../../../data/seshat_cross_version_pmt_baseline_ordered \
    --validation_path ../../../data/seshat_cross_version_pmt_baseline_ordered \
    --model_config ../../../src/model_configs/pmt_baseline.json\
    --sent_vocab_path ../../../data/seshat_cross_version_pmt_baseline_ordered \
    --body_vocab_path ../../../data/seshat_cross_version_pmt_baseline_ordered \
    --project Lang \
    2>&1 | tee log.txt