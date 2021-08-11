open_retrieval_dir=$1
mode=$2
cmd=$3

for lang in arabic bengali english finnish indonesian japanese korean russian swahili telugu thai
do
	echo $lang
	if [ "$mode" = "eval" ]; then
		sh eval_single.sh $open_retrieval_dir $lang $cmd
	elif [ "$mode" = "default" ]; then
		sh run_bm25_single_default.sh $open_retrieval_dir $lang
	else
		python run_bm25_single.py $open_retrieval_dir $lang
	fi
done
