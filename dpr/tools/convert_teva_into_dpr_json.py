""" This file prepare the train / dev .json file that could be fit into the https://github.com/luyug/GC-DPR """
import os
import sys
import json
import glob

from tqdm import tqdm

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os_dir(os.path.abspath(__file__))))
sys.path.append(PACKAGE_PATH)

from utils import load_runs, load_qrels, load_topic_tsv, load_collection_jsonl 



def format_json_entry(tevatron_entry):
    query_id = tevatron_entry["query_id"]
    query = tevatron_entry["query"]
    # print(tevatron_entry.keys())
    positive_passages = tevatron_entry["positive_passages"] if "positive_passages" in tevatron_entry else []
    negative_passages = tevatron_entry["negative_passages"] if "negative_passages"in tevatron_entry else []
 
    return {
        "question": query,
        "answers": [],
        "positive_ctxs": [
            {"title": "", "text": doc["title"] + doc["text"]} for doc in positive_passages
        ],
        "negative_ctxs": [],
        "hard_negative_ctxs": [
            {"title": "", "text": doc["title"] + doc["text"]} for doc in negative_passages
        ],
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/convert_teva_into_dpr_json.py /path/to/teva.jsonl")
        exit()

    version = "v1.1"
    teva_jsonl = sys.argv[1]
    assert teva_jsonl.endswith(".jsonl")
    output_json = teva_jsonl[:-6] + ".gcdpr.json"

    gcdpr_training_data = [format_json_entry(json.loads(line)) for line in open(teva_jsonl)]
    json.dump(gcdpr_training_data, open(output_json, "w"), ensure_ascii=False)


if __name__ == "__main__":
	main()
