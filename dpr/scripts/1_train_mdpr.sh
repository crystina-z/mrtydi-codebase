#!/bin/bash
#SBATCH -p gpu20
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH --time 48:00:00
#SBATCH --output outputs/dpr-mbert-cased.with-validation.train.log

nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

# set -e

cd GC-DPR

# pretrain_model=bert-base-multilingual-uncased
pretrain_model=bert-base-multilingual-cased

python train_dense_encoder.py \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg $pretrain_model \
   --train_file "dpr_data/data/mrtydi/train/*train.json" \
   --dev_file "dpr_data/data/mrtydi/dev/*dev.json" \
   --output_dir outputs \
   --grad_cache \
   --q_chunk_size 16 \
   --ctx_chunk_size 8 \
   --fp16  # if train with mixed precision

cd ..
