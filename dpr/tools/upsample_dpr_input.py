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

dpr_input_dir = sys.argv[1]
N_Q = 12373  # i.e. number of arabic


for lang in sorted(LANGS):
    dpr_input_fn = os.path.join(dpr_input_dir, lang, "train.gcdpr.json")
    outp_dpr_input_fn = os.path.join(dpr_input_dir, lang, f"train.gcdpr-NQ_{N_Q}.json")
    lines = json.load(open(dpr_input_fn))
    n_lines = len(lines) 
    quotient = N_Q // n_lines
    remainder = N_Q - n_lines * quotient 
    upsampled_lines = lines * quotient + random.choices(lines, k=remainder)
    print(lang, "Ori: ", len(lines), "upsampled: ", len(upsampled_lines))
    assert len(upsampled_lines) == N_Q

    json.dump(upsampled_lines, open(outp_dpr_input_fn, "w"), ensure_ascii=False)