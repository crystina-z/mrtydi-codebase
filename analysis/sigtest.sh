for lang in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai  
do
	echo "---------------------------------------------"
	echo $lang
	for metric in recip_rank ndcg_cut_10
	do
	echo $metric
		bm25_tuned=$(ls runfiles/bm25/$lang/*bm25.test.k1*)
		python -m nirtools.ir.sig_test \
			--runfile1 $bm25_tuned \
			--runfile2 runfiles/DPR/run.hybrid.test.$lang.trec \
			--qrels ../dataset/open-retrieval/$lang/qrels.test.txt \
			--metric $metric
	done
done
