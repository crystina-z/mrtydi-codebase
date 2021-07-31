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
import matplotlib.pyplot as plt

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


def interpolate(bm25, mdpr, interpolate_qids, interpolate_k=0, allow_mdpr_doc=False):
    """
    bm25 and mdpr are two run objects, with the format {qid-0: {docid-0: score}, qid-1: {}...};
    interpolate_qids: a set of qids to consider in the interpolation;
    interpolate_k: only conduct interpolation on the top k documents;
    allow_mdpr_doc: 
        only used when interpolate_k > 0, indicating while we are using hybrid score for the top-k bm25, 
        do we allow to add docids in mdpr but not in *entire* bm25 list to be added in this list.
        Note that this means when adding *n* mdpr docids to top-k, *n* bm25 documents would be squeezed off 
        (i.e. we lose them forever).
    """
    interpolated = {}
    for qid in bm25:
        if (qid not in interpolate_qids) or (qid not in mdpr):
            interpolated[qid] = deepcopy(normalize(bm25[qid]))
            continue

        normalized_bm25 = normalize(bm25[qid])
        normalized_mdpr = normalize(mdpr[qid])
        all_docids = set(normalized_bm25) | set(normalized_mdpr)
        interpolated_docid2scores = {
            docid: (normalized_bm25.get(docid, 0) + normalized_mdpr.get(docid, 0) / 2) for docid in all_docids
        }
        if interpolate_k > 0:  # only use the new score for the top100 documents; and make sure they are always top100
            top_k_bm25 = keep_topk(normalized_bm25, k=interpolate_k)

            if not allow_mdpr_doc:
                topk_interpolated = {docid: interpolated_docid2scores[docid] + 1 if docid in top_k_bm25 else normalized_bm25[docid] for docid in normalized_bm25}
            else:
                topk_interpolated = {
                    docid: interpolated_docid2scores[docid] + 1 if ((docid in top_k_bm25) or (docid not in normalized_bm25)) else normalized_bm25[docid] 
                    for docid in interpolated_docid2scores
                }  # consider all docids rather than just the ones from bm25
                topk_interpolated = keep_topk(topk_interpolated, k=interpolate_k)

                if interpolate_k == 1000:
                    assert set(topk_interpolated) == set(keep_topk(interpolated_docid2scores, k=1000)) # sanity check

            interpolated_docid2scores = topk_interpolated

        # keep only the top 1k
        interpolated[qid] = keep_topk(interpolated_docid2scores, k=1000)
    return interpolated


def stringify(dct):
    kv = sorted(dct.items())
    # return " ".join([f"{k}: %.4f" % v for k, v in kv])
    return "\t".join([f"%.4f" % v for k, v in kv])


def interpolate_main(bm25, mdpr, qrels, criterion, metrics, interpolate_top_k, allow_mdpr_doc):
    qid2recall1k = evaluate(qrels, runs=bm25, metrics={criterion}, aggregate=False)
    qid_pos_recall1k = {qid for qid in qid2recall1k if qid2recall1k[qid][criterion] > 0}
    qid_zero_recall1k = {qid for qid in qid2recall1k if qid2recall1k[qid][criterion] == 0} | (set(mdpr) - set(bm25))
    all_queries = set(bm25) | set(mdpr)
    assert len(qid_zero_recall1k) + len(qid_pos_recall1k) == len(all_queries)

    # for tag, qids_to_interpolate in zip(
    #     ["ALL", "R@k>0", "R@k ==0"],
    #     [all_queries, qid_pos_recall1k, qid_zero_recall1k]
    # ):
    interpolated = interpolate(bm25, mdpr, qid_pos_recall1k, interpolate_k=interpolate_top_k, allow_mdpr_doc=allow_mdpr_doc)
    score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)
    # print(f"{tag + ' [ ' + str(len(qids_to_interpolate)) + ' ]':20}", stringify(score))
    return score


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)
    qrels = load_qrels(args.qrels_file)

    # all_k = [10, 20, 50, 100, 1000]
    all_k = [10, 20, 50, 100]
    criterion = args.criterion
    # interpolate_top_k = args.interpolate_top_k
    # allow_mdpr_doc = args.allow_mdpr_doc

    # print(f"{'':20}NDCG@10\tMRR")
    all_queries = set(bm25) | set(mdpr)
 
    # for tag, runs in zip(["BM25", "mDPR"], [bm25, mdpr]):
    bm25_score = evaluate(qrels, bm25, metrics=metrics, aggregate=True)
    mdpr_score = evaluate(qrels, mdpr, metrics=metrics, aggregate=True)
    interpolated = interpolate(bm25, mdpr, all_queries, interpolate_k=1000, allow_mdpr_doc=True)
    hybrid_score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)

    metric = "ndcg_cut_10"
    plt.plot([0, all_k[-1]], [bm25_score[metric]] * 2, label="BM25", color="tab:pink", linestyle="--")
    # plt.plot([0, 1000], [mdpr_score[metric]] * 2, label="mDPR")
    plt.plot([0, all_k[-1]], [hybrid_score[metric]] * 2, label="Hybrid", color="tab:grey", linestyle="--")

    for allow_mdpr_doc in [True, False]: 
        kwargs = {"linestyle": "--" if allow_mdpr_doc else "-"}
        label = "Add mdpr doc" if allow_mdpr_doc else "No mdpr doc" 
        scores = [
            interpolate_main(bm25, mdpr, qrels, criterion, metrics, interpolate_top_k=k, allow_mdpr_doc=allow_mdpr_doc)[metric]
            for k in all_k
        ]
        plt.scatter(all_k, scores, color="tab:blue", s=10)
        plt.plot(all_k, scores, label=label, color="tab:blue", **kwargs)

    plt.legend()
    plt.xticks(ticks=all_k, labels=list(map(str, all_k)))
    plt.savefig("test.png")


def main_q(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)
    qrels = load_qrels(args.qrels_file)

    metric = "ndcg_cut_10"
    # all_k = [10, 20, 50, 100, 1000]
    all_k = [10, 20, 50, 100]
    # interpolate_top_k = args.interpolate_top_k
    # allow_mdpr_doc = args.allow_mdpr_doc

    # print(f"{'':20}NDCG@10\tMRR")
    all_queries = set(bm25) | set(mdpr)
 
    # for tag, runs in zip(["BM25", "mDPR"], [bm25, mdpr]):
    bm25_score = evaluate(qrels, bm25, metrics=metrics, aggregate=True)
    mdpr_score = evaluate(qrels, mdpr, metrics=metrics, aggregate=True)
    interpolated = interpolate(bm25, mdpr, all_queries, interpolate_k=1000, allow_mdpr_doc=True)
    hybrid_score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)

    plt.plot([0, all_k[-1]], [bm25_score[metric]] * 2, label="BM25", color="tab:pink", linestyle="--")
    # plt.plot([0, 1000], [mdpr_score[metric]] * 2, label="mDPR")
    plt.plot([0, all_k[-1]], [hybrid_score[metric]] * 2, label="Hybrid", color="tab:grey", linestyle="--") 

    # criterion = args.criterion
    for allow_mdpr_doc in [True, False]: 
        kwargs = {"linestyle": "--" if allow_mdpr_doc else "-"}
        label = "Add mdpr doc" if allow_mdpr_doc else "No mdpr doc" 
        scores = [
            interpolate_main(bm25, mdpr, qrels, f"recall_{k}", metrics, interpolate_top_k=100, allow_mdpr_doc=allow_mdpr_doc)[metric]
            for k in all_k
        ]
        plt.scatter(all_k, scores, color="tab:blue", s=10)
        plt.plot(all_k, scores, label=label, color="tab:blue", **kwargs)

    plt.legend()
    plt.xticks(ticks=all_k, labels=list(map(str, all_k)))
    plt.savefig("varying-criteria.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bm25-runfile", "-bm25", type=str, required=True)
    parser.add_argument("--mdpr-runfile", "-mdpr", type=str, required=True)
    parser.add_argument("--qrels-file", "-q", type=str, required=True)

    parser.add_argument("--criterion", "-c", type=str, default="recall_1000")
    parser.add_argument("--interpolate-top-k", "-k", type=int, default=0, help="If not zero, then only the top k documents in bm25 would be interpolated with mDPR.")
    parser.add_argument("--allow-mdpr-doc", action="store_true", default=False, help="Only used when interpolate-top-k > 0.") 

    parser.add_argument("--output-dir", "-o", type=str, default="./tmp-interpolation", help="If not given, a ./tmp-interpolation folder would be created.")

    args = parser.parse_args()

    main(args)
    # main_q(args)
