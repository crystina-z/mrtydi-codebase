""" 
This file sample from the traini.json files produced by `generate_dpr_json.py` so that all languages have same number of queries. 
For now 800 queries is preserved for each language, according to the stats prepared by `print_dpr_input_size.py`
"""
import os
import sys
import json
import glob
import random

import numpy as np
from tqdm import tqdm

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os_dir(os.path.abspath(__file__))))
sys.path.append(PACKAGE_PATH)

from utils import LANGS, load_runs, load_qrels, load_topic_tsv, load_collection_jsonl 

random.seed(123)
N_Q = 800


def main():
    dpr_input_dir = sys.argv[1]
    for lang in LANGS:
        json_fn = os_join(dpr_input_dir, lang, "train.json")
        output_json_fn = os_join(dpr_input_dir, lang, "train.sampled.json")
        assert os.path.exists(json_fn)

        dpr_inputs = json.load(open(json_fn))
        dpr_inputs_preserved = random.choices(dpr_inputs, k=N_Q)
        print(lang, ":", len(dpr_inputs_preserved), "out of", len(dpr_inputs), "queries are preserved")
        assert len(dpr_inputs_preserved) == N_Q
        json.dump(dpr_inputs_preserved, open(output_json_fn, "w"))


if __name__ == "__main__":
    main()
