""" 
This file explore the three different interpolation method:
- bm25 + mdpr
- bm25 + (mdpr with the queries whose bm25 recall > 0)
- bm25 + (mdpr with the queries whose bm25 recall == 0)
"""
import os
import sys
from copy import deepcopy
from argparse import ArgumentParser 

import numpy as np
import pytrec_eval

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_dir))
from utils import load_runs, load_qrels

import pdb

# metrics = {"ndcg_cut_10", "recall_10", "recall_1000", "recip_rank"}
metrics = {"ndcg_cut_10", "recip_rank"}


def evaluate(qrels, runs, metrics, aggregate):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    eval_results = evaluator.evaluate(runs)
    scores = {
        qid: [metrics_dict.get(m, -1) for m in metrics] for qid, metrics_dict in eval_results.items()
    }

    if aggregate: 
        scores = list(scores.values())
        scores = np.array(scores).mean(axis=0).tolist()
        scores = dict(zip(metrics, scores))
    else:
        scores = {qid: dict(zip(metrics, metric_scores)) for qid, metric_scores in scores.items()}

    return scores


def normalize(doc2score):
    """ scale all scores from orignal range to (0, 1) """
    min_score, max_score = min(doc2score.values()), max(doc2score.values())
    ori_range = max_score - min_score
    if ori_range == 0: # all docid hv same score 
        assert len(set(doc2score.values())) == 1
        return {docid: 0.5 for docid in doc2score}

    return {docid: (score - min_score) / ori_range for docid, score in doc2score.items()}


def keep_topk(doc2score, k=1000):
    docid_scores = sorted(doc2score.items(), key=lambda kv: float(kv[1]), reverse=True)[:k]
    return {docid: score for docid, score in docid_scores}


def interpolate(bm25, mdpr, qids_to_interpolate):
    interpolated = {}
    for qid in bm25:
        if (qid not in qids_to_interpolate) or (qid not in mdpr):
            interpolated[qid] = deepcopy(normalize(bm25[qid]))
            continue

        normalized_bm25 = normalize(bm25[qid])
        normalized_mdpr = normalize(mdpr[qid])
        all_docids = set(normalized_bm25) | set(normalized_mdpr)
        interpolated_docid2scores = {
            docid: (normalized_bm25.get(docid, 0) + normalized_mdpr.get(docid, 0) / 2) for docid in all_docids
        }
        # keep only the top 1k
        interpolated[qid] = keep_topk(interpolated_docid2scores, k=1000)
    return interpolated


def stringify(dct):
    kv = sorted(dct.items())
    # return " ".join([f"{k}: %.4f" % v for k, v in kv])
    return "\t".join([f"%.4f" % v for k, v in kv])


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)
    qrels = load_qrels(args.qrels_file)
    criterion = args.criterion

    qid2recall1k = evaluate(qrels, runs=bm25, metrics={criterion}, aggregate=False)
    all_queries = set(bm25) | set(mdpr)
    qid_pos_recall1k = {qid for qid in qid2recall1k if qid2recall1k[qid][criterion] > 0}
    qid_zero_recall1k = {qid for qid in qid2recall1k if qid2recall1k[qid][criterion] == 0} | (set(mdpr) - set(bm25))

    assert len(qid_zero_recall1k) + len(qid_pos_recall1k) == len(all_queries)

    # print(f"{'':20}NDCG@10\tR@10\tR@1000\tMRR")
    print(f"{'':20}NDCG@10\tMRR")

    for tag, runs in zip(["BM25", "mDPR"], [bm25, mdpr]):
        score = evaluate(qrels, runs, metrics=metrics, aggregate=True)
        print(f"{tag:20}", stringify(score))

    for tag, qids_to_interpolate in zip(
        ["ALL", "R@k>0", "R@k ==0"],
        [all_queries, qid_pos_recall1k, qid_zero_recall1k]
    ):
        interpolated = interpolate(bm25, mdpr, qids_to_interpolate)
        score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)
        print(f"{tag + ' [ ' + str(len(qids_to_interpolate)) + ' ]':20}", stringify(score))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bm25-runfile", "-bm25", type=str, required=True)
    parser.add_argument("--mdpr-runfile", "-mdpr", type=str, required=True)
    parser.add_argument("--qrels-file", "-q", type=str, required=True)

    parser.add_argument("--criterion", "-c", type=str, default="recall_1000")

    parser.add_argument("--output-dir", "-o", type=str, default="./tmp-interpolation", help="If not given, a ./tmp-interpolation folder would be created.")

    args = parser.parse_args()
    main(args)
