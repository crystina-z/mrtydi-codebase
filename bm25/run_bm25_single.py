import os
import sys
import json
import shutil

import pytrec_eval
from pprint import pprint
 
os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(__file__))
sys.path.append(PACKAGE_PATH)

from constants import *
from utils import str_list, load_runs, load_qrels, run_command, aggregate_score


if len(sys.argv) != 3:
    print("Usage: python run_bm25_single.py /path/to/open_retrieval_dir lang")
    exit()

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


def search_fn(
        index_path, topic_fn, topicreader, runfile, lang_abbr, 
        k1_s, b_s, hits,
        run_rm3=False, fb_terms=None, fb_docs=None, ori_weights=None,
    ):
    """ Execute bm25 or bm25 + rm3; when no rm3 parameter is passed, use the default Anserini one """
    rm3_params = []
    if run_rm3:
        rm3_params.append("-rm3") 
        if fb_terms is not None:
            assert isinstance(fb_terms, list) or isinstance(fb_terms, tuple)
            rm3_params.extend(["-rm3.fbTerms", *str_list(fb_terms)])

        if fb_docs is not None:
            assert isinstance(fb_docs, list) or isinstance(fb_docs, tuple)
            rm3_params.extend(["-rm3.fbDocs", *str_list(fb_docs)])

        if ori_weights is not None:
            assert isinstance(ori_weights, list) or isinstance(ori_weights, tuple)
            rm3_params.extend(["-rm3.originalQueryWeight", *str_list(ori_weights)])

    lang_params = ["-language", lang_abbr] if lang_abbr not in ["sw", "te"] else ["-pretokenized"]
    cmd = [search, 
        "-index", index_path, 
        "-topics", topic_fn, "-topicreader", topicreader,
        "-output", runfile, 
        "-hits", str(hits),
        "-bm25", "-bm25.k1", *str_list(k1_s), "-bm25.b", *str_list(b_s), 
        "-threads", "16", 
        *lang_params,
        *rm3_params,
    ]
    run_command(cmd)


def tune_parameters(k1_s, b_s, hits, is_rm3=False, fb_terms=None, fb_docs=None, ori_weights=None):
    set_name = "dev" 
    topic_fn = f"{root_dir}/topic.{set_name}.tsv"
    runfile = f"{runfile_dir}/bm25{'rm3' if is_rm3 else ''}.{set_name}"
    donefile = f"{runfile_dir}/done.bm25{'rm3' if is_rm3 else ''}"

    # fb_terms = [None] if fb_terms is None else fb_terms
    # fb_docs = [None] if fb_docs is None else fb_docs 
    # ori_weights = [None] if ori_weights is None else ori_weights 
    def gen_paras():
        if not is_rm3:
            return [(k1, b) for k1 in k1_s for b in b_s]
        else:
            assert isinstance(fb_terms, list) and isinstance(fb_docs, list) and isinstance(ori_weights, list)
            return [
                (k1, b, fb_term, fb_doc, ori_weight)
                for k1 in k1_s for b in b_s 
                for fb_term in fb_terms for fb_doc in fb_docs for ori_weight in ori_weights
            ]


    # tune parameters on dev set
    if not os.path.exists(donefile):
        try:
            search_fn(
                index_path, topic_fn, topicreader, runfile, lang_abbr, 
                k1_s=k1_s, b_s=b_s, hits=hits,
                run_rm3=is_rm3, fb_terms=fb_terms, fb_docs=fb_docs, ori_weights=ori_weights,
            ) 
        except ValueError as e:
            raise e
        else:
            with open(donefile, "w") as f:
                f.write("done\n")

    # evaluate and find the best parameters
    best = [-1, {}, {}]  # (score (optimize), {all metric: score}, {para: value}) 
    best_json_fn = f"{runfile_dir}/best.bm25{'rm3' if is_rm3 else ''}.json"
    if os.path.exists(best_json_fn):
        best = json.load(open(best_json_fn))
        # return best[1], best[2]
    else:
        qrels_fn = f"{root_dir}/qrels.{set_name}.txt"
        qrels = load_qrels(qrels_fn)
        for params in gen_paras():
            # k1, b if is_rm3 else k1, b, fb_term, fb_doc, ori_weight)
            expected_n_params = 5 if is_rm3 else 2
            assert len(params) == expected_n_params

            evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", "ndcg_cut"})
            cur_runfile = f"{runfile}_bm25(k1=%.1f,b=%.1f)_default" if not is_rm3 else \
                f"{runfile}_bm25(k1=%.1f,b=%.1f)_rm3(fbTerms=%d,fbDocs=%d,originalQueryWeight=%.1f)"
            cur_runfile = cur_runfile % params 

            runs = load_runs(cur_runfile)
            qid2score = evaluator.evaluate(runs)
            metric2score = aggregate_score(qid2score)
            if metric2score[optimize] > best[0]:
                best = (
                    metric2score[optimize], 
                    metric2score, 
                    # {"k1": k1, "b": b},
                    dict(zip(["k1", "b", "fb_term", "fb_doc", "ori_weight"], params)),
                )

    print("best score:")
    pprint(best[1])
    # k1, b = best[2]["k1"], best[2]["b"]
    # if is_rm3:
    #     fb_term, fb_doc, ori_weight = best[2]["fb_term"], best[2]["fb_doc"], best[2]["ori_weight"]
    params = best[2]
    suffix = ".".join([f"{k}={v}" for k, v in params.items()])

    for set_name in ["train", "test"]:
        topic_fn = f"{root_dir}/topic.{set_name}.tsv"
        # runfile = f"{runfile_dir}/bm25.{set_name}.k1={k1}.b1={b}" if not is_rm3 else \
        #     f"{runfile_dir}/bm25rm3.{set_name}.k1={k1}.b1={b}.fb_term={fb_term}.fb_doc={fb_doc}.ori_weight={ori_weight}"
        runfile = f"{runfile_dir}/bm25{'rm3' if is_rm3 else ''}.{set_name}.{suffix}"
        if not os.path.exists(runfile):
            if is_rm3:
                search_fn(
                    index_path, topic_fn, topicreader, runfile, lang_abbr, 
                    [params["k1"]], [params["b"]], hits,
                    True, [params["fb_term"]], [params["fb_doc"]], [params["ori_weight"]],
                ) 
            else:
                search_fn(
                    index_path, topic_fn, topicreader, runfile, lang_abbr, 
                    [params["k1"]], [params["b"]], hits,
                )

    # json.dump(best, open(f"{runfile_dir}/best.bm25{'rm3' if is_rm3 else ''}.json", "w"))
    json.dump(best, open(best_json_fn, "w"))
    return best[1], params 


# index 
if not os.path.exists(index_path):
    lang_params = ["-language", lang_abbr] if lang_abbr not in ["sw", "te"] else ["-pretokenized"]
    cmd = [index, 
        "-collection", collection_type,
        "-input", collection_dir,
        "-index", index_path,
        "-generator", "DefaultLuceneDocumentGenerator",
        "-threads", "16", "-storePositions", "-storeDocvectors", "-storeRaw", 
        *lang_params,
    ]
    run_command(cmd)


# tune parameters for bm25 and rm3
print("running bm25")
score, params = tune_parameters(    # bm25
    k1_s=k1_s, b_s=b_s, hits=hits, is_rm3=False)

print("running bm25rm3")
score, params = tune_parameters(    # rm3 (does not tune k1 and b to save time and space)
    k1_s=[params["k1"]], b_s=[params["b"]], hits=hits, is_rm3=True, 
    fb_terms=fb_terms, fb_docs=fb_docs, ori_weights=ori_weights)
