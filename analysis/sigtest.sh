#!/bin/bash

dataset_dir="/store/scratch/x978zhan/mr-tydi/dataset"
results_dir="/store/scratch/x978zhan/mr-tydi/results"
mode=$1

echo "bm25 tuned vs hybrid"
for lang in arabic bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
	echo "---------------------------------------------"
	echo $lang
	for metric in recip_rank recall_1000
	do
		echo $metric
		bm25_tuned=$(ls $results_dir/bm25/runs/${lang}/bm25.test.k1*)
		if [[ $mode == "em" ]]; then
			qrel_fn=${dataset_dir}/../qid2answers/${lang}/test.em.uniq.qrel
		else
			qrel_fn=${dataset_dir}/$lang/qrels.test.txt
		fi

		echo "Using qrel fn: " $qrel_fn
		python -m nirtools.ir.sig_test \
			--runfile1 $bm25_tuned \
			--runfile2 ${results_dir}/hybrid/runs/$lang/run.hybrid.test.$lang.trec \
			--qrels $qrel_fn \
			--metric $metric
	done
done


exit

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

