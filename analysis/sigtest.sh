dataset_dir="/store/scratch/x978zhan/mr-tydi/dataset"
results_dir="/store/scratch/x978zhan/mr-tydi/results"

echo "bm25 vs bm25 tuned"
for lang in arabic bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
	echo "---------------------------------------------"
	echo $lang
	for metric in recip_rank ndcg_cut_10
	do
		echo $metric
		bm25_default=$(ls $results_dir/bm25/runs/$lang/bm25.test.default*)
		bm25_tuned=$(ls $results_dir/bm25/runs/$lang/bm25.test.k1*)

		python -m nirtools.ir.sig_test \
			--runfile1 $bm25_default \
			--runfile2 $bm25_tuned \
			--qrels ${dataset_dir}/$lang/qrels.test.txt \
			--metric $metric
	done
done


echo "bm25 tuned vs hybrid"
for lang in arabic bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
	echo "---------------------------------------------"
	echo $lang
	for metric in recip_rank ndcg_cut_10
	do
		echo $metric
		bm25_tuned=$(ls $results_dir/bm25/runs/$lang/bm25.test.k1*)

		python -m nirtools.ir.sig_test \
			--runfile1 $bm25_tuned \
			--runfile2 ${results_dir}/hybrid/run.hybrid.test.$lang.trec \
			--qrels ${dataset_dir}/$lang/qrels.test.txt \
			--metric $metric
	done
done
