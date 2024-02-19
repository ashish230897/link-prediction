BASE="$PWD"
MODEL_TYPE="roformer"

OUT_DIR_NAME=$1

OUT_DIR="$BASE/model/${OUT_DIR_NAME}"
mkdir -p $OUT_DIR

MLM_TRAIN_FILE="${BASE}/data/wordnet18rr/text/train.txt"
MLM_EVAL_FILE="${BASE}/data/wordnet18rr/text/valid.txt"

export CUDA_VISIBLE_DEVICES='0'

BATCH_SIZE=128
GRAD_STEPS=1
LEARNING_RATE=3e-5
MAX_STEPS=100000
WARMUP=1000
SAVE_STEPS=2000
EVAL_STEPS=2000

if [ ! -d $OUT_DIR ] 
then
  mkdir -p $OUT_DIR
fi

python $BASE/code/transformer_mlm.py \
    --model_type $MODEL_TYPE \
    --output_dir $OUT_DIR \
    --do_train \
    --do_eval \
    --logging_strategy 'steps'\
    --logging_dir './logs/'\
    --logging_steps 10\
    --seed 42 \
    --overwrite_output_dir \
    --dropout_rate 0.1 \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type 'constant_with_warmup' \
    --weight_decay 0.01 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --remove_unused_columns False \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy 'steps'\
    --prediction_loss_only True \
    --wandb_project 'link-pred-pretraining' \
    --train_data_file $MLM_TRAIN_FILE \
    --eval_data_file $MLM_EVAL_FILE \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --gradient_accumulation_steps $GRAD_STEPS \
    --report_to wandb \