import os
import sys
import json
import shutil

import pytrec_eval
from pprint import pprint
 
from constants import *
from utils import str_list, load_runs, load_qrels, run_command, aggregate_score


open_retrieval_dir = sys.argv[1]
lang = sys.argv[2]
root_dir = f"{open_retrieval_dir}/{lang}"

if not os.path.exists(root_dir):
    print(f"unfound directory {root_dir}")
    exit()

# convert lang into the anserini version
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
if not os.path.exists(index_path):
    cmd = [index, 
        "-collection", collection_type,
        "-input", collection_dir,
        "-index", index_path,
        "-language", lang_abbr,
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "16", "-storePositions", "-storeDocvectors", "-storeRaw", 
    ]
    run_command(cmd)


def search_fn(
        index_path, topic_fn, topicreader, runfile, lang_abbr, 
        k1_s, b_s, hits,
        run_rm3=False, fb_terms=None, fb_docs=None, ori_weight=None,
    ):
    """ Execute bm25 or bm25 + rm3; when no rm3 parameter is passed, use the default Anserini one """
    rm3_params = []
    if run_rm3:
        rm3_params.append("-rm3") 
        if fb_terms is not None:
            assert isinstance(fb_terms, list) or isinstance(fb_terms, tuple)
            rm3_params.extends(["-rm3.fbTerms", *str_list(fb_terms)])

        if fb_docs is not None:
            assert isinstance(fb_docs, list) or isinstance(fb_docs, tuple)
            rm3_params.extends(["-rm3.fbDocs", *str_list(fb_docs)])

        if ori_weight is not None:
            assert isinstance(ori_weight, list) or isinstance(ori_weight, tuple)
            rm3_params.extends(["-rm3.originalQueryWeight", *str_list(ori_weight)])

    cmd = [search, 
        "-index", index_path, 
        "-topics", topic_fn, "-topicreader", topicreader,
        "-output", runfile, 
        "-language", lang_abbr, 
        "-hits", str(hits),
        "-bm25", "-bm25.k1", *str_list(k1_s), "-bm25.b", *str_list(b_s), 
        "-threads", "16", 
    ] + rm3_params
    run_command(cmd)


set_name = "dev" 
topic_fn = f"{root_dir}/topic.{set_name}.tsv"
runfile = f"{runfile_dir}/bm25.{set_name}"
donefile = f"{runfile_dir}/done"

# tune parameters on dev set
if not os.path.exists(donefile):
    try:
        search_fn(index_path, topic_fn, topicreader, runfile, lang_abbr, k1_s, b_s, hits) 
    except ValueError as e:
        raise e
    else:
        with open(donefile, "w") as f:
            f.write("done\n")


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
