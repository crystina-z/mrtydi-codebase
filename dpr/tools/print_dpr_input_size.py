""" This file prepare the training .json file that could be fit into the https://github.com/luyug/GC-DPR """
import os
import sys
import json
import glob

import numpy as np
from tqdm import tqdm

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os_dir(os.path.abspath(__file__))))
sys.path.append(PACKAGE_PATH)

from utils import LANGS, load_runs, load_qrels, load_topic_tsv, load_collection_jsonl 


def main():
    dpr_input_dir = sys.argv[1]
    # lang = sys.argv[2]
    print(f"{'lang':20}{'No. Q':20}{'No. Pos (avg)':20}{'No. Neg (avg)':20}")
    for lang in LANGS:
        json_fn = os_join(dpr_input_dir, lang, "train.json")
        assert os.path.exists(json_fn)

        dpr_inputs = json.load(open(json_fn))
        n_q = len(dpr_inputs)
        n_pos_s = [len(entry["positive_ctxs"]) for entry in dpr_inputs]
        n_neg_s = [len(entry["hard_negative_ctxs"]) for entry in dpr_inputs]
        print(f"{lang:20}{n_q:20}{round(np.mean(n_pos_s), 2):20}{round(np.mean(n_neg_s), 2):20}")


if __name__ == "__main__":
    main()
