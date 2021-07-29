for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
    echo "========================================"
    echo $lang
    echo "========================================"

    python interpolate.py \
        -q ../data/${lang}/qrels.test.txt \
        -bm25 ../runfiles/bm25/$lang/bm25.test.k1*.b* \
        -mdpr ../runfiles/DPR/run.dense.test.${lang}.trec
    echo 
    echo 
done
