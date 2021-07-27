#!/bin/bash
#SBATCH -p gpu20
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --gres gpu:4
#SBATCH --time 12:00:00
#SBATCH --output outputs/index-search.dpr-mbert-uncased.ft-on-mrtydi.bengali.log

# cd pyserini

nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

hf_model_dir="./models/hf-models"
mrtydi_data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi"


lang=bengali

test_topic_fn="${mrtydi_data_dir}/ori/${lang}/topic.test.tsv"
coll_json_dir="${mrtydi_data_dir}/test/${lang}_collection_jsonl"



# files to output
index_dir="${mrtydi_data_dir}/faiss_index/$lang"
runfile="runs/${lang}.run.mdpr.mrtydi-test/trec"


# 1. index
if [ ! -d $index_dir ]; then
    mkdir -p $index_dir/..
    python -m pyserini.dindex \
        --encoder "$hf_model_dir/mdpr-context-encoder" \
        --corpus $coll_json_dir \
        --index $index_dir
else
    echo "Found existing ${index_dir}, skip."
fi


# 2. search
if [ ! -f $runfile ]; then
    python -m pyserini.dsearch \
        --topics $test_topic_fn \
        --index $index_dir \
        --encoder "$hf_model_dir/mdpr-question-encoder" \
        --output $runfile \
        --batch-size 36 --threads 12
else
    echo "Found existing ${runfile}, skip."
fi


# cd ..
