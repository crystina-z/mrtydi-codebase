import os
import sys
import json
import shutil
import subprocess

import numpy as np
import pytrec_eval
from pprint import pprint
 
from utils import *


open_retrieval_dir = sys.argv[1]
lang = sys.argv[2]
ANSERINI_DIR = os.environ["ANSERINI_DIR"]

optimize = "recip_rank"
root_dir = f"{open_retrieval_dir}/{lang}"
index = f"{ANSERINI_DIR}/target/appassembler/bin/IndexCollection"
search = f"{ANSERINI_DIR}/target/appassembler/bin/SearchCollection"

if not os.path.exists(root_dir):
    print(f"unfound directory {root_dir}")
    exit()

# convert lang into the anserini version
lang_abbr=""
lang2abbr = {
    "english": "en",
    "finnish": "fi",
    "japanese": "ja",
    "thai": "th",
    "russian": "ru",
    "arabic": "ar",
    "bengali": "bn",
    "indonesian": "in",
    "korean": "ko",
}
lang_abbr = lang2abbr[lang] 

# output directories
collection_dir=f"{root_dir}/collection"
collection_file=f"{root_dir}/collection/collection.txt"
index_path=f"{root_dir}/index/lucene-index.pos+docvectors+raw"
runfile_dir=f"{root_dir}/runfiles"

for dir in [collection_dir, os.path.dirname(index_path), runfile_dir]:
    os.makedirs(dir, exist_ok=True)

# prepare collection
if not os.path.exists(collection_file):
   shutil.copyfile(f"{root_dir}/collection.txt", collection_file)

# index 
collection_type = "TrecCollection"
if not os.path.exists(index_path):
    cmd = [index, 
        "-collection", collection_type,
        "-input", collection_dir,
        "-index", index_path,
        "-language", lang_abbr,
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "16", "-storePositions", "-storeDocvectors", "-storeRaw", 
    ]
    rtn = subprocess.run(cmd) 
    if rtn.returncode != 0:
        print(" ".join(cmd) + "\tCommand Failed") 
        raise ValueError("returncode of java cmd is not 0")


# search (train and dev) 
def str_list(lst):
    return list(map(str, lst)) 


def search_fn(index_path, topic_fn, topicreader, runfile, lang_abbr, k1_s, b_s, hits):
    cmd = [search, 
        "-index", index_path, 
        "-topics", topic_fn, "-topicreader", topicreader,
        "-output", runfile, 
        "-language", lang_abbr, 
        "-bm25", "-bm25.k1", *str_list(k1_s), "-bm25.b", *str_list(b_s), "-hits", str(hits),
        "-threads", "16", 
    ]

    rtn = subprocess.run(cmd) 
    if rtn.returncode != 0:
        print(" ".join(cmd) + "\tCommand Failed") 
        raise ValueError("returncode of java cmd is not 0")


hits = 1000
topicreader = "TsvString"

set_name = "dev" 
topic_fn = f"{root_dir}/topic.{set_name}.tsv"
runfile = f"{runfile_dir}/bm25.{set_name}"
donefile = f"{runfile_dir}/done"

# tune parameters on dev set
k1_s = [float("%.2f" % v) for v in np.arange(0.4, 1.6, 0.1)]
b_s = [float("%.2f" % v) for v in np.arange(0.1, 1.0, 0.1)]


if not os.path.exists(donefile):
    try:
        search_fn(index_path, topic_fn, topicreader, runfile, lang_abbr, k1_s, b_s, hits) 
    except ValueError as e:
        raise e
    else:
        with open(donefile, "w") as f:
            f.write("done\n")


def aggregate_score(qid2score):
    aggregated = defaultdict(list)
    for qid in qid2score:
        for name, value in qid2score[qid].items():
            aggregated[name].append(value)         

    for name in aggregated:
        aggregated[name] = np.mean(aggregated[name])

    return aggregated


# evaluate
best = [-1, {}, {}]  # (score (optimize), {all metric: score}, {para: value}) 
qrels_fn = f"{root_dir}/qrels.{set_name}.txt"
qrels = load_qrels(qrels_fn)
for k1 in k1_s:
    for b in b_s:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", "ndcg_cut"})
        cur_runfile = f"{runfile}_bm25(k1=%.1f,b=%.1f)_default" % (k1, b)
        runs = load_runs(cur_runfile)
        qid2score = evaluator.evaluate(runs)
        metric2score = aggregate_score(qid2score)
        if metric2score[optimize] > best[0]:
            best = (metric2score[optimize], metric2score, {"k1": k1, "b": b})

print("best score:")
pprint(best[1])
k1, b = best[2]["k1"], best[2]["b"]

for set_name in ["train", "test"]:
    topic_fn = f"{root_dir}/topic.{set_name}.tsv"
    runfile = f"{runfile_dir}/bm25.{set_name}.k1={k1}.b1={b}"
    if not os.path.exists(runfile):
        search_fn(index_path, topic_fn, topicreader, runfile, lang_abbr, [k1], [b], hits) 


json.dump(best, open(f"{runfile_dir}/best.json", "w"))
