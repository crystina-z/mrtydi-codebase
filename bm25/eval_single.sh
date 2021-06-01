open_retrieval_dir=$1
lang=$2
trec_cmd=$3  # e.g. -J, -c, etc

root_dir="${open_retrieval_dir}/${lang}"
runfile_dir="${root_dir}/runfiles"

# search (train and dev) 
k1=0.9
b=0.4
hits=1000
topicreader="TsvString"

for set_name in "train" "dev"
do
    echo "==============\n" $set_name "\n=============="
    qrels_fn="${root_dir}/qrels.${set_name}.txt"
    runfile="${runfile_dir}/bm25.${set_name}.k1=$k1.b=$b.txt"
    if [ ! -f $runfile ]; then 
	echo "unfound $runfile"
	exit
    fi

    if [ ! -f $qrels_fn ]; then 
	echo "unfound $qrels_fn"
	exit
    fi

    trec_eval $trec_cmd $qrels_fn $runfile | head -n 5
    echo 

    trec_eval $trec_cmd $qrels_fn $runfile -m map -m P.1,5,10,20 -m ndcg_cut.10,20
    echo 
done
