root_dir="/store/scratch/x978zhan/mr-tydi/v1.1"
dataset_dir="${root_dir}/dataset"
results_dir="${root_dir}/results"

for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
    echo "========================================"
    echo $lang
    echo "========================================"

    bm25_fn=${results_dir}/bm25-runfiles-top100/$lang/bm25.test.k1*.b*

    python hist_of_ranks.py \
        -l $lang --tag dense \
        -q ${dataset_dir}/mrtydi-v1.1-${lang}/qrels.test.txt \
        -bm25 ${bm25_fn} \
        -mdpr ${results_dir}/mdpr-runs/${lang}/test.trec.top100

    python hist_of_ranks.py \
        -l $lang --tag hybrid \
        -q ${dataset_dir}/mrtydi-v1.1-${lang}/qrels.test.txt \
        -bm25 ${bm25_fn} \
        -mdpr ${results_dir}/hybrid-runs/${lang}/hybrid.${lang}.zeroshot.top100

    echo 
done
