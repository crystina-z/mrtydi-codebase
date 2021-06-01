data_output_dir=$1
if [ -z $data_output_dir ]; then
	echo "Requried to provide directory path to store the dataset"
	echo "Example: sh prepare_dataset.sh /path/to/output_dir"
	exit
fi

mkdir -p $data_output_dir

wiki_dir="${data_output_dir}/Wiki"
tydi_dir="${data_output_dir}/TyDi"
open_retrieval_dir="${data_output_dir}/open-retrieval"

# download and extract the raw dataset
sh 0_download_and_extract.tydi.sh $tydi_dir
sh 0_download_and_extract.wiki.sh $wiki_dir

# extract jsonl lines containing valid answers
sh 1_categorized_tydi_data.sh $tydi_dir


# generate topics.txt, qrels.txt, collections, etc.
python 2_prepare_open_retrieval_tydi.py \
	--tydi_dir "${tydi_dir}/with_answer" \
	--wiki_dir $wiki_dir \
	--output_dir $open_retrieval_dir


# split topics.tsv and qrels.txt into train and dev set
python 3_split_train_dev_set.py $open_retrieval_dir


echo "finished"