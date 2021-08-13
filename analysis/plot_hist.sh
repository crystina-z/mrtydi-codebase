dataset_dir="/store/scratch/x978zhan/mr-tydi/dataset"
results_dir="/store/scratch/x978zhan/mr-tydi/results"

for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
    echo "========================================"
    echo $lang
    echo "========================================"

    # for tag in dense hybrid
    # do
    python hist_of_ranks.py \
	-l $lang --tag dense \
	-q ${dataset_dir}/${lang}/qrels.test.txt \
	-bm25 ${results_dir}/bm25/runs/$lang/bm25.test.k1*.b* \
	-mdpr ${results_dir}/mdpr/runs/$lang/test.trec

    python hist_of_ranks.py \
	-l $lang --tag hybrid \
	-q ${dataset_dir}/${lang}/qrels.test.txt \
	-bm25 ${results_dir}/bm25/runs/$lang/bm25.test.k1*.b* \
	-mdpr ${results_dir}/hybrid/run.hybrid.test.${lang}.trec 

    # done
    echo 
done

