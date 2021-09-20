for dir in arabic  bengali  english  finnish  indonesian  japanese  korean  russian  swahili  telugu  thai 
do 
	echo $dir 
	cat models/dpr_outputs/nq/hf_format/results/runs/$dir/dev.trec | cut -d ' ' -f 1 | sort | uniq > tmp-qid-mdpr
	cat models/dpr_outputs/nq/hf_format/results/bm25-runfiles/$dir/bm25.dev.k1* | cut -d ' ' -f 1 | sort | uniq > tmp-qid-bm25
	diff tmp-qid-mdpr tmp-qid-bm25
done
