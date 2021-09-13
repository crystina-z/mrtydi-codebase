open_retrieval_dir=$1
lang=$2
trec_cmd=$3  # e.g. -J, -c, etc

root_dir="${open_retrieval_dir}/*${lang}"
runfile_dir="${open_retrieval_dir}/../bm25-runfiles/${lang}"
# runfile_dir="${root_dir}/runfiles"

# for set_name in "train" "test"
for set_name in "test"
do
    echo "==============\n" $set_name "\n=============="
    qrels_fn="${root_dir}/qrels.${set_name}.txt"
    for runfile in ${runfile_dir}/bm25*.${set_name}.*
    do
	    echo $runfile
	    if [ ! -f $(ls $runfile) ]; then
		echo "unfound $runfile"
		exit
	    fi

	    if [ ! -f $qrels_fn ]; then
		echo "unfound $qrels_fn"
		exit
	    fi

	    trec_eval $trec_cmd $qrels_fn $runfile | head -n 5
	    echo

	    # trec_eval $trec_cmd $qrels_fn $runfile  -m recip_rank -m recall.1000
	    trec_eval -c $trec_cmd $qrels_fn $runfile  -m recip_rank -m recall.100,1000
	    echo
   done
done
