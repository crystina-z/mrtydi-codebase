#!/bin/bash

dataset_dir="/store/scratch/x978zhan/mr-tydi/v1.1/dataset"
results_dir="/store/scratch/x978zhan/mr-tydi/v1.1/results"
mode=$1

echo "bm25 tuned vs hybrid"
for lang in arabic bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai
do
	echo "---------------------------------------------"
	echo $lang
	for metric in recip_rank recall_100
	do
		echo $metric
		# bm25_tuned=$(ls $results_dir/bm25-runfiles-top100/${lang}/bm25.test.k1*)
		bm25_tuned="/store/scratch/x978zhan/mr-tydi/v1.1/bm25-runfiles/${lang}/bm25.test.k1*"
		# if [[ $mode == "em" ]]; then
		# 	qrel_fn=${dataset_dir}/../qid2answers/${lang}/test.em.uniq.qrel
		# else
		qrel_fn=${dataset_dir}/mrtydi-v1.1-$lang/qrels.test.txt
		# fi

		hybrid=${results_dir}/hybrid-runs/$lang/hybrid.$lang.zeroshot.top100
		wc -l $bm25_tuned 
		wc -l $hybrid

		echo "Using qrel fn: " $qrel_fn
		echo "BM25" && trec_eval $qrel_fn $bm25_tuned -m recip_rank -m recall.100 
		echo "Hybrid" && trec_eval $qrel_fn $hybrid -m recip_rank -m recall.100 
		python -m nirtools.ir.sig_test \
			--runfile1 $bm25_tuned \
			--runfile2 $hybrid \
			--qrels $qrel_fn \
			--metric $metric
	done
done

