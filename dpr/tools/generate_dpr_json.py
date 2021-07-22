""" This file prepare the training .json file that could be fit into the https://github.com/luyug/GC-DPR """
import os
import sys
import json
import glob

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os_dir(os_dir(__file__))))
sys.path.append(PACKAGE_PATH)

from utils import load_runs, load_qrels, load_topic_tsv, load_collection_trec


def get_train_runfile(dir):
	fns = [fn for fn in glob.glob(f"{dir}/bm25.*train*") if "default" not in fn]
	assert len(fns) == 1, f"Expect only one runfile, but got {len(fns)}: {fns}"
	return fns[0]


def format_json_entry(query, pos_docids, neg_docids, id2doc):
	""" 
	- skip answers field;
	- put all neg_docids into `hard_negative_ctxs` field;
	- leave title field empty
	"""
	return {  
		"question": query,
		"answers": [],
		"positive_ctxs": [
			{"title": "", "text": id2doc[pos_docid]} for pos_docid in pos_docids
		],
		"negative_ctxs": [],
		"hard_negative_ctxs": [
			{"title": "", "text": id2doc[neg_docid]} for neg_docid in neg_docids
		], 
	}


def main():
	if len(sys.argv) != 3:
		print("Usage: python run_bm25_single.py /path/to/open_retrieval_dir lang")
		exit()

	open_retrieval_dir = sys.argv[1]
	lang = sys.argv[2]
	lang_dir = os_join(open_retrieval_dir, lang) 

	topic_fn = os_join(lang_dir, "topic.tsv")
	folds_fn = os_join(lang_dir, "folds.json")
	qrels_fn = os_join(lang_dir, "qrels.txt")
	coll_fn = os_join(lang_dir, "collection", "collection.txt")
	runfile_dir = os_join(lang_dir, "runfiles")
	train_runfile = get_train_runfile(runfile_dir)
	assert all(map(os.exists, [topic_fn, folds_fn, qrels_fn, coll_fn]))
	dpr_dir = os_join(lang_dir, "dpr_inputs")
	output_json = os_join(dpr_dir, "train.json")

	folds = json.load(open(folds_fn))
	qrels = load_qrels(qrels_fn)
	train_runs = load_runs(train_runfile)

	id2query = load_topic_tsv(topic_fn)
	id2doc = {id: doc for id, doc in load_collection_trec(coll_fn)}

	# todo: add support to dev and test set
	n_unfound = 0
	training_data = []
	for qid, query in id2query.items():
		if qid not in folds["train"]:
			continue
		
		if qid not in train_runs:
			print(f"Warning: {qid} could not be found in runfile")
			continue

		pos_docids = [docid for docid in train_runs[qid] if qrels.get(docid, 0) > 0]
		neg_docids = [docid for docid in train_runs[qid] if qrels.get(docid, 0) == 0]
		if len(pos_docids) == 0:
			n_unfound += 1
			continue

		json_entry = format_json_entry(query, pos_docids, neg_docids, id2doc)
		training_data.append(json_entry)
	
	json.dump(training_data, open(output_json, "w"))
	print(
		"Finished.",
		f"{len(training_data)} queries have been stored in {output_json}."
		f"{n_unfound} queries do not have positive doc found in {train_runfile}"
	)
