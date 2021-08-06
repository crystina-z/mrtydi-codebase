"""
This file plots the histogram of the first relevant document return by bm25 and mdpr
"""
import os
import sys
from copy import deepcopy
from argparse import ArgumentParser 

import numpy as np
import pytrec_eval
import matplotlib
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_dir))
from utils import load_runs, load_qrels

file_dir = os.path.dirname(os.path.abspath(__file__)) 
plot_dir = f"{file_dir}/plots"

import pdb

def _hist(data, bins, ax=None, **kwargs):
    if not ax:
        """ to plot negative histgram"""
        fig = plt.Figure()
        ax = matplotlib.axes.Axes(fig, (0,0,0,0))
        n, _, patches = ax.hist(x=data, bins=bins)
        del ax, fig
    else:
        n, _, patches = ax.hist(x=data, bins=bins, **kwargs)

    return n, patches


def plot_hist(data, bins, ax, direction="pos", **kwargs):
    assert direction in {"pos", "neg"}
    if direction == "pos":  # normal plots
        n, patches = _hist(data, bins, ax=ax, **kwargs)
    elif direction == "neg":  # normal plots
        n = _hist(data, bins)
        # pdb.set_trace()
        assert len(n) == len(bins) - 1
        neg_n = [-value for value in n]
        n, patches = _hist(data, bins, ax=ax, bottom=neg_n, **kwargs)  # plot the histogram at the neg axis
    return n, patches 


def plot_ratio(data, x_s, ax, **kwargs):
    sum_data = sum(data) 
    normalized_data = [x / sum_data for x in data]
    return ax.plot(x_s, normalized_data, marker="x", **kwargs)
    # ax.scatter(x_s, normalized_data, marker="x", **kwargs)


def get_rank_list(qrels, runs):
    """ return a list of rank, -1 indicates the rel doc could not be found"""
    ranks = []
    for qid, doc2score in runs.items():
        if qid not in qrels:
            raise ValueError(f"{qid} could not be found.")

        pos_docids = {docid for docid in qrels[qid] if qrels[qid][docid] > 0}
        if not pos_docids: 
            raise ValueError(f"No positive doc found for {qid}.")

        cur_ranks = [i for i, (docid, score) in enumerate(
            sorted(doc2score.items(), key=lambda kv: kv[1], reverse=True)
        ) if docid in pos_docids]
        cur_rank = -1 if len(cur_ranks) == 0 else min(cur_ranks)
        ranks.append(cur_rank)
    return ranks


def main(args):
    tag = args.tag
    mdpr_label = "mDPR" if tag == "dense" else "Hybrid"

    qrels = load_qrels(args.qrels_file)
    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)

    bm25_ranks = get_rank_list(qrels, bm25)
    mdpr_ranks = get_rank_list(qrels, mdpr)

    fig = plt.figure()
    ax = plt.gca()


    common_args = {"alpha": 0.3}
    bins = [-10] + list(range(0, 110, 10))
    bm25_n, bm25_patches = plot_hist(bm25_ranks, bins=bins, ax=ax, direction="pos", label="BM25", color="tab:blue", **common_args)
    mdpr_n, mdpr_patches = plot_hist(mdpr_ranks, bins=bins, ax=ax, direction="pos", label=mdpr_label, color="tab:orange", **common_args)

    plt.xticks(bins, ["Unfound"] + bins[1:])
    # plt.legend()

    plt.grid(color="lightgray")
    plt.xlabel("Top-k documents")
    plt.ylabel("No. of Q with relevant document")
    lang = args.lang.capitalize()

    # plot the right axis
    ax_right = ax.twinx()
    x_s = [
        (bins[i] + bins[i + 1]) / 2 for i in range(1, len(bins) - 1)
    ]
    common_args = {"alpha": 0.9}
    bm25_ratio_patches = plot_ratio(bm25_n[1:], x_s, ax=ax_right, color="tab:blue", label="BM25 (ratio)", **common_args)
    mdpr_ratio_patches = plot_ratio(mdpr_n[1:], x_s, ax=ax_right, color="tab:orange", label=f"{mdpr_label} (ratio)", **common_args)
    ax_right.set_ylabel("% of Queries among all queries with D_rel found")

    # combined all labels
    all_patches = bm25_patches.patches + mdpr_patches.patches + bm25_ratio_patches + mdpr_ratio_patches
    labels = [l.get_label() for l in all_patches]
    ax_right.legend(all_patches, labels, loc=0)

    plt.title(lang)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/rank-hist/{tag}/rank-hist-{lang}-{tag}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bm25-runfile", "-bm25", type=str, required=True)
    parser.add_argument("--mdpr-runfile", "-mdpr", type=str, required=True)
    parser.add_argument("--qrels-file", "-q", type=str, required=True)
    parser.add_argument("--lang", "-l", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--output-dir", "-o", type=str, default="./tmp-interpolation", help="If not given, a ./tmp-interpolation folder would be created.")

    args = parser.parse_args()

    main(args)
