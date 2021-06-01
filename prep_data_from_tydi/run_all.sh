mode=$1
cmd=$2

for lang in thai finnish bengali russian japanese arabic indonesian korean english 
do
	echo $lang
	if [ $mode = "eval" ]; then
		sh eval_single.sh $lang $cmd
	else
		sh run_bm25_single.sh $lang
	fi
done
