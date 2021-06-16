open_retrieval_dir=$1
mode=$2
cmd=$3

for lang in thai finnish bengali russian japanese arabic indonesian korean english
do
	echo $lang
	if [ "$mode" = "eval" ]; then
		sh bm25/eval_single.sh $open_retrieval_dir $lang $cmd

	elif [ "$mode" = "rm3" ]; then
		sh bm25/run_bm25rm3_single.sh $open_retrieval_dir $lang

	else
		python bm25/run_bm25_single.py $open_retrieval_dir $lang
	fi
done
