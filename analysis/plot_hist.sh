for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
    echo "========================================"
    echo $lang
    echo "========================================"

    for tag in dense hybrid
    do
	    # python hist_of_ranks.py -q ../data/english/qrels.test.txt -bm25 ../runfiles/bm25/english/bm25.test.k1* -mdpr ../runfiles/DPR/run.hybrid.test.english.trec  -l english  
	    python hist_of_ranks.py \
		-l $lang --tag $tag \
		-q ../data/${lang}/qrels.test.txt \
		-bm25 ../runfiles/bm25/$lang/bm25.test.k1*.b* \
		-mdpr ../runfiles/DPR/run.${tag}.test.${lang}.trec 
    done
    echo 
done

