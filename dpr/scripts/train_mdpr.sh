set -e

cd GC-DPR

pretrain_model=bert-base-multilingual-uncased

python train_dense_encoder.py \
   --encoder_model_type hf_bert \
   --pretrained_model_cfg $pretrain_model \
   --train_file "dpr_data/data/mrtydi/*json" \
   --dev_file "dpr_data/data/mrtydi/*json" \
   --output_dir outputs \
   --grad_cache \
   --q_chunk_size 16 \
   --ctx_chunk_size 8 \
   --fp16  # if train with mixed precision

cd ..