for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
# for lang in arabic   #english  finnish  korean  
do
    echo "========================================"
    echo $lang
    echo "========================================"

    python interpolate.py \
	--lang $lang \
        -q ../data/${lang}/qrels.test.txt \
        -bm25 ../runfiles/bm25/$lang/bm25.test.k1*.b* \
        -mdpr ../runfiles/DPR/run.dense.test.${lang}.trec # \
        # -c recall_1000 \
        # -k 1000 \
        # --allow-mdpr-doc
    echo 
    echo 
done
