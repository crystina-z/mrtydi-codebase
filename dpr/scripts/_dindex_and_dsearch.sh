"""
This file does not run by itself, need the $hf_model_line provide by scripts/3_dindex_and_dsearch.sh
"""
nvidia-smi
source /GW/NeuralIR/work/cuda-10.1_env.sh

version="v1.1"
# todo, rename. this contain the same contents with open-retrieval
# mrtydi_data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/mrtydi"
# mrtydi_data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/v0.6"
mrtydi_data_dir="/GW/carpet/nobackup/czhang/dpr/data/mrtydi/${version}"

results_dir="$hf_model_dir/results"
index_dir="${results_dir}/faiss_index"
run_dir="${results_dir}/runs"
mkdir -p $run_dir

if [ ! -d $hf_model_dir ]; then
	echo "Cannot find $hf_model_dir"
fi

for lang in bengali telugu finnish swahili thai indonesian arabic korean japanese russian english
do
    lang_data_dir="${mrtydi_data_dir}/mrtydi-${version}-${lang}"
    coll_json_dir="${lang_data_dir}/collection"

    # files to output
    lang_index_dir="${index_dir}/${lang}"

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

    for set_name in "dev" "test"
    do
        # topic_fn="${mrtydi_data_dir}/${lang}/topic.${set_name}.tsv"
        topic_fn="${lang_data_dir}/topic.${set_name}.tsv"
        runfile="${run_dir}/${lang}/${set_name}.trec"

        if [ ! -f $runfile ]; then
            python -m pyserini.dsearch \
                --topics $topic_fn \
                --index $lang_index_dir \
                --encoder "$hf_model_dir/mdpr-question-encoder" \
                --output $runfile \
                --batch-size 36 --threads 12
        else
            echo "Found existing ${runfile}, skip."
        fi
    done
done
