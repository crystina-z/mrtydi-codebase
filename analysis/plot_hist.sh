dataset_dir="/store/scratch/x978zhan/mr-tydi/v0.6/dataset"
results_dir="/store/scratch/x978zhan/mr-tydi/v0.6/results"

for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
    echo "========================================"
    echo $lang
    echo "========================================"

    bm25_fn=${results_dir}/bm25/runs-top100/$lang/bm25.test.k1*.b*

    python hist_of_ranks.py \
	-l $lang --tag dense \
	-q ${dataset_dir}/${lang}/qrels.test.txt \
	-bm25 ${bm25_fn} \
	-mdpr ${results_dir}/mdpr/runs-top100/${lang}/test.trec*

    python hist_of_ranks.py \
	-l $lang --tag hybrid \
	-q ${dataset_dir}/${lang}/qrels.test.txt \
	-bm25 ${bm25_fn} \
	-mdpr ${results_dir}/hybrid/runs-top100/${lang}/run.hybrid.test.${lang}.trec*

    echo 
done
