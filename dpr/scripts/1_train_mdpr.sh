#!/bin/bash
#SBATCH -p gpu20 
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 12:00:00
#SBATCH --output outputs/mrtydi-v1.1-delimiter-nn-dpr-mbert-cased.with-validation.train-sampled-data.727-th.log

nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

set -e

cd GC-DPR

version="v1.1"

# exclude_lang=arabic
# exclude_lang=bengali
# exclude_lang=english
# exclude_lang=finnish
# exclude_lang=indonesian
# exclude_lang=japanese
# exclude_lang=korean
# exclude_lang=russian 
# exclude_lang=swahili
# exclude_lang=telugu
exclude_lang=thai
# exclude_lang="nothing-to-exclude"

# pretrain_model=bert-base-multilingual-uncased
pretrain_model=bert-base-multilingual-cased
# output_dir="outputs/mytydi-${version}.mbert-cased.validation.sampled-data.no-${exclude_lang}"
# output_dir="outputs/mytydi-${version}.mbert-cased.validation.sampled-data-727"
# output_dir="outputs/mytydi-v1.1-delimiter-nn.mbert-cased.validation.sampled-data-727"
# output_dir="outputs/mytydi-v1.1-delimiter-nn.mbert-cased.validation"
output_dir="outputs/mytydi-v1.1-delimiter-nn.mbert-cased.validation.no-${exclude_lang}"
mkdir -p $output_dir

# batch_size=128
batch_size=16

# data_dir="dpr_data/data/mrtydi"
# data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/${version}/dpr_inputs"
data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/v1.1-delimiter-nn/dpr_inputs"

# train_file_pattern="$data_dir/*/train.sampled.800.json"
train_file_pattern="$data_dir/*/train.sampled.727.json"
# train_file_pattern="$data_dir/*/train.json"

python train_dense_encoder.py \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg $pretrain_model \
   --train_file "$train_file_pattern" \
   --dev_file  "$data_dir/*/dev.json" \
   --exclude_str $exclude_lang \
   --output_dir $output_dir \
   --grad_cache \
   --batch_size $batch_size \
   --q_chunk_size 16 \
   --ctx_chunk_size 8 \
   --fp16  # if train with mixed precision

cd ..
