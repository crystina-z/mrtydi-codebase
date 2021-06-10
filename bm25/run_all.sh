open_retrieval_dir=$1
mode=$2
cmd=$3

for lang in thai finnish bengali russian japanese arabic indonesian korean english 
do
	echo $lang
	if [ "$mode" = "eval" ]; then
		sh bm25/eval_single.sh $open_retrieval_dir $lang $cmd
	else
		sh bm25/run_bm25_single.sh $open_retrieval_dir $lang
	fi
done
