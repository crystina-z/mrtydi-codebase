"""
This file does not run by itself, need the $hf_model_line provide by scripts/3_dindex_and_dsearch.sh
"""
nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

# todo, rename. this contain the same contents with open-retrieval
mrtydi_data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/mrtydi"

results_dir="$hf_model_dir/results"
index_dir="${results_dir}/faiss_index"
run_dir="${results_dir}/runs"
mkdir -p $run_dir

if [ ! -d $hf_model_dir ]; then
	echo "Cannot find $hf_model_dir"
fi

for lang in bengali telugu finnish swahili thai indonesian arabic korean japanese russian english
do
    test_topic_fn="${mrtydi_data_dir}/${lang}/topic.test.tsv"
    coll_json_dir="${mrtydi_data_dir}/${lang}/collection"

    # files to output
    lang_index_dir="${index_dir}/${lang}"
    runfile="${run_dir}/${lang}/trec"

    # 1. index
    if [ ! -f "${lang_index_dir}/index" ]; then
        mkdir -p ${lang_index_dir}
        python -m pyserini.dindex \
            --encoder "$hf_model_dir/mdpr-context-encoder" \
            --corpus $coll_json_dir \
            --index $lang_index_dir
    else
        echo "Found existing ${lang_index_dir}/index, skip."
    fi

    # 2. search
    if [ ! -f $runfile ]; then
        python -m pyserini.dsearch \
            --topics $test_topic_fn \
            --index $lang_index_dir \
            --encoder "$hf_model_dir/mdpr-question-encoder" \
            --output $runfile \
            --batch-size 36 --threads 12
    else
        echo "Found existing ${runfile}, skip."
    fi

done
