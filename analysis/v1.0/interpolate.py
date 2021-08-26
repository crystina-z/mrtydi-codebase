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

metrics = {"ndcg_cut_10", "recip_rank"}
file_dir = os.path.dirname(os.path.abspath(__file__)) 
plot_dir = f"{file_dir}/plots"


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


def interpolate(bm25, mdpr, interpolate_qids=None, topk=0, mdpr_doc_mode="doc_and_score"):
    """
    bm25 and mdpr are two run objects, with the format {qid-0: {docid-0: score}, qid-1: {}...};
    interpolate_qids: 
        a set of qids to consider in the interpolation;
        use all if interpolate_qids is empty
    topk: only conduct interpolation on the topk documents;
    # add_mdpr_doc: 
    #     only used when interpolate_k > 0, indicating while we are using hybrid score for the top-k bm25, 
    #     do we allow to add docids in mdpr but not in *entire* bm25 list to be added in this list.
    #     Note that this means when adding *n* mdpr docids to top-k, *n* bm25 documents would be squeezed off 
    #     (i.e. we lose them forever).
    mdpr_doc_mode:
        doc_and_score: use both the new doc and the score from mdpr (standard hybrid)
        score_hybrid: do not use the new doc found by mdpr, and mdpr score is "averaged" with bm25 score
        score_replace: do not use the new doc found by mdpr, and mdpr score is used to replaced with bm25 score
    """
    interpolated = {}
    if not interpolate_qids:
        interpolate_qids = set(bm25) | set(mdpr)

    # for qid in bm25:
    for qid in interpolate_qids:
        if qid not in bm25:
            interpolated[qid] = deepcopy(normalize(mdpr[qid]))
            continue

        if qid not in mdpr:
            interpolated[qid] = deepcopy(normalize(bm25[qid]))
            continue

        # qid is in both bm25 and mdpr
        normalized_bm25 = normalize(bm25[qid])
        normalized_mdpr = normalize(mdpr[qid])
        all_docids = set(normalized_bm25) | set(normalized_mdpr)
        interpolated_docid2scores = {
            docid: (normalized_bm25.get(docid, 0) + normalized_mdpr.get(docid, 0) / 2) for docid in all_docids
        }

        if topk > 0:  # only use the new score for the top100 documents; and make sure they are always top100
            if mdpr_doc_mode == "doc_and_score":  # consider all docids rather than just the ones from bm25
                top_k_docids = keep_topk(interpolated_docid2scores, k=topk)
                topk_interpolated = {}
                for docid in interpolated_docid2scores:
                    if docid in top_k_docids:
                        topk_interpolated[docid] = interpolated_docid2scores[docid] + 1
                    elif docid in normalized_bm25:
                        topk_interpolated[docid] = normalized_bm25[docid]
                    # do not store the doc if it's either not in top-k and not in bm25

                topk_interpolated = keep_topk(topk_interpolated, k=topk)
                if topk == 1000:
                    assert set(topk_interpolated) == set(keep_topk(interpolated_docid2scores, k=1000)) # sanity check
            else:
                top_k_docids = keep_topk(normalized_bm25, k=topk)  # use only the bm25 docs
                if mdpr_doc_mode == "score_hybrid":
                    topk_interpolated = {
                        docid: interpolated_docid2scores[docid] + 1 if docid in top_k_docids else normalized_bm25[docid] 
                        for docid in normalized_bm25
                    }
                elif mdpr_doc_mode == "score_replace":
                    topk_interpolated = {
                        docid: normalized_mdpr[docid] + 1 if ((docid in top_k_docids and docid in normalized_mdpr)) else normalized_bm25[docid] 
                        for docid in normalized_bm25
                    }
                else:
                    raise ValueError(f"Unexpected mode {mdpr_doc_mode}.")

                # don't need to re-trunk the docids into top-1k since we only use the ones from bm25

            interpolated_docid2scores = topk_interpolated

        # keep only the top 1k
        interpolated[qid] = keep_topk(interpolated_docid2scores, k=1000)
    return interpolated


# def interpolate_main(bm25, mdpr, qrels, criterion, metrics, topk, add_mdpr_doc):
def interpolate_main(bm25, mdpr, qrels, criterion, metrics, topk, mdpr_doc_mode):
    # qid2recall1k = evaluate(qrels, runs=bm25, metrics={criterion}, aggregate=False)
    interpolated = interpolate(
        bm25,
        mdpr, 
        interpolate_qids=None, 
        topk=topk, 
        # add_mdpr_doc=add_mdpr_doc
        mdpr_doc_mode=mdpr_doc_mode
    )
    score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)
    return score


def main(args):
    lang = args.lang
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)
    qrels = load_qrels(args.qrels_file)

    # all_k = [10, 20, 50, 100, 1000]
    # all_k = [10, 20, 50, 100, 100]
    all_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    criterion = args.criterion
    # interpolate_top_k = args.interpolate_top_k
    # allow_mdpr_doc = args.allow_mdpr_doc

    # for tag, runs in zip(["BM25", "mDPR"], [bm25, mdpr]):
    bm25_score = evaluate(qrels, bm25, metrics=metrics, aggregate=True)
    mdpr_score = evaluate(qrels, mdpr, metrics=metrics, aggregate=True)
    # interpolated = interpolate(bm25, mdpr, None, topk=1000, add_mdpr_doc=True)
    interpolated = interpolate(bm25, mdpr)
    hybrid_score = evaluate(qrels, interpolated, metrics=metrics, aggregate=True)

    # metric = "ndcg_cut_10"
    metric = "recip_rank"
    plt.plot([0, all_k[-1]], [bm25_score[metric]] * 2, label="BM25", color="tab:pink", linestyle="--")
    # plt.plot([0, 1000], [mdpr_score[metric]] * 2, label="mDPR")
    plt.plot([0, all_k[-1]], [hybrid_score[metric]] * 2, label="Hybrid", color="tab:grey", linestyle="--")

    # for add_mdpr_doc in [True, False]: 
    for mdpr_doc_mode in ["doc_and_score", "score_hybrid", "score_replace"]:
        # kwargs = {"linestyle": "--" if add_mdpr_doc else "-"}
        # label = "Add mdpr doc" if add_mdpr_doc else "No mdpr doc" 
        if mdpr_doc_mode == "doc_and_score":
            label = "Add mdpr doc"
            kwargs = {"linestyle": "-"}

        elif mdpr_doc_mode == "score_hybrid":
            label = "No mdpr doc (hybrid score)"
            kwargs = {"linestyle": "--"}

        elif mdpr_doc_mode == "score_replace":
            label = "No mdpr doc (mdpr score only)"
            kwargs = {"linestyle": "-."}

        scores = [
            interpolate_main(bm25, mdpr, qrels, criterion, metrics, topk=k, mdpr_doc_mode=mdpr_doc_mode)[metric]
            for k in all_k
        ]
        plt.scatter(all_k, scores, color="tab:blue", s=10)
        plt.plot(all_k, scores, label=label, color="tab:blue", **kwargs)

    plt.legend()
    plt.xticks(ticks=all_k, labels=list(map(str, all_k)))

    plt.savefig(f"{plot_dir}/{lang}-{metric}.png")
    # plt.savefig(".png")


def base_score(qrels, runs, criterion="recall_1000"):
    """ calculate the standard score and the score where R@1k > 0 """
    base_score = evaluate(qrels, runs, metrics=metrics, aggregate=True)

    qid2cri_scores = evaluate(qrels, runs=runs, metrics={criterion}, aggregate=False)
    qids_positive_cri = {q for q, scores in qid2cri_scores.items() if scores[criterion] > 0}
    runs_positive_cri = {q: doc2score for q, doc2score in runs.items() if q in qids_positive_cri}
    pos_r1k_score = evaluate(qrels, runs_positive_cri, metrics=metrics, aggregate=True)

    print(base_score)
    print(pos_r1k_score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--bm25-runfile", "-bm25", type=str, required=True)
    parser.add_argument("--mdpr-runfile", "-mdpr", type=str, required=True)
    parser.add_argument("--qrels-file", "-q", type=str, required=True)

    parser.add_argument("--criterion", "-c", type=str, default="recall_1000")
    # parser.add_argument("--interpolate-top-k", "-k", type=int, default=0, help="If not zero, then only the top k documents in bm25 would be interpolated with mDPR.")
    # parser.add_argument("--allow-mdpr-doc", action="store_true", default=False, help="Only used when interpolate-top-k > 0.") 
    parser.add_argument("--output-dir", "-o", type=str, default="./tmp-interpolation", help="If not given, a ./tmp-interpolation folder would be created.")

    args = parser.parse_args()

    # main(args)
    base_score(qrels=load_qrels(args.qrels_file), runs=load_runs(args.bm25_runfile))
    base_score(qrels=load_qrels(args.qrels_file), runs=load_runs(args.mdpr_runfile))
