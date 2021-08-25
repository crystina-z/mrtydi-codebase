#!/bin/bash
#SBATCH -p gpu20
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 48:00:00
#SBATCH --output outputs/mrtydi-v1.1-dpr-mbert-cased.with-validation.train.log

nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

set -e

cd GC-DPR

# pretrain_model=bert-base-multilingual-uncased
pretrain_model=bert-base-multilingual-cased

# data_dir="dpr_data/data/mrtydi"
data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/v1.1/dpr_inputs"

python train_dense_encoder.py \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg $pretrain_model \
   --train_file "$data_dir/*/train.json" \
   --dev_file "$data_dir/*/dev.json" \
   --output_dir outputs \
   --grad_cache \
   --q_chunk_size 16 \
   --ctx_chunk_size 8 \
   --fp16  # if train with mixed precision

cd ..
