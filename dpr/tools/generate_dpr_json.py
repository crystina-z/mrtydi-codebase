""" This file prepare the training .json file that could be fit into the https://github.com/luyug/GC-DPR """
import os
import sys
import json
import glob

from tqdm import tqdm

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os_dir(os.path.abspath(__file__))))
sys.path.append(PACKAGE_PATH)

from utils import load_runs, load_qrels, load_topic_tsv, load_collection_trec


def get_runfile(dir, mode):
	assert mode in {'train', 'dev'} # we do not need jsonl for test set

	fns = [fn for fn in glob.glob(f"{dir}/bm25.*{mode}*") if "default" not in fn]
	if len(fns) == 0:
		import pdb
		pdb.set_trace()
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

	assert all(map(os.path.exists, [topic_fn, folds_fn, qrels_fn, coll_fn]))
	folds = json.load(open(folds_fn))
	qrels = load_qrels(qrels_fn)
	id2query = load_topic_tsv(topic_fn)
	id2doc = {id: doc for id, doc in load_collection_trec(coll_fn)}

	dpr_dir = os_join(lang_dir, "dpr_inputs")

	for set_name in ["train", "dev"]:
		output_json = os_join(dpr_dir, f"{set_name}.json")
		if os.path.exists(output_json):
			print(f"Found existing {output_json}, skip.")
			continue

		os.makedirs(dpr_dir, exist_ok=True)

		train_runfile = get_runfile(runfile_dir, mode=set_name)
		train_runs = load_runs(train_runfile)
		# todo: add support to dev and test set
		n_unfound = 0
		training_data = []
		for qid, query in tqdm(id2query.items(), desc=f"{lang} ({set_name})"):
			if qid not in folds[set_name]:
				continue
		
			if qid not in train_runs:
				print(f"Warning: {qid} could not be found in runfile")
				continue

			pos_docids = [docid for docid in train_runs[qid] if qrels[qid].get(docid, 0) > 0]
			neg_docids = [docid for docid in train_runs[qid] if qrels[qid].get(docid, 0) == 0]

			import pdb
			# pdb.set_trace()

			if len(pos_docids) == 0:
				n_unfound += 1
				continue

			json_entry = format_json_entry(query, pos_docids, neg_docids, id2doc)
			training_data.append(json_entry)
	
		json.dump(training_data, open(output_json, "w"))
		print(
			"Finished.",
			f"{len(training_data)} queries have been stored in {output_json}.",
			f"{n_unfound} queries do not have positive doc found in {train_runfile}"
		)



if __name__ == "__main__":
	main()
