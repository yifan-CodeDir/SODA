
GPUID="0,1,2,3"

# CUDA_VISIBLE_DEVICES=$GPUID python run.py \
## accelerate training
export CUDA_VISIBLE_DEVICES=$GPUID 
accelerate launch --num_processes 4 --config_file ./acc_config.yaml train.py \
    --pretrained_trans codet5 \
    --experiment cross_project_soda_fold4 \
    --train_path ../../../data/cross_project_soda_fold4 \
    --pos_path ../../../data/cross_project_soda_fold4 \
    --validation_path ../../../data/cross_project_soda_fold4 \
    --model_config ../../../src/model_configs/soda_trans_test.json\
    --is_diff \
    2>&1 | tee log.txt