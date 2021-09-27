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

os_dir = os.path.dirname
file_dir = os_dir(os.path.abspath(__file__))
sys.path.append(os_dir(os_dir(file_dir)))
from utils import load_runs, load_qrels

file_dir = os_dir(os.path.abspath(__file__)) 
plot_dir = f"{file_dir}/plots"


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
        assert len(n) == len(bins) - 1
        neg_n = [-value for value in n]
        n, patches = _hist(data, bins, ax=ax, bottom=neg_n, **kwargs)  # plot the histogram at the neg axis
    return n, patches 


def plot_ratio(data, x_s, ax, **kwargs):
    sum_data = sum(data) 
    normalized_data = [x / sum_data for x in data]
    return ax.plot(x_s, normalized_data, marker="x", **kwargs)


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
        cur_rank = 101 if len(cur_ranks) == 0 else min(cur_ranks)
        ranks.append(cur_rank)
    return ranks


def main(args):
    tag = args.tag
    mdpr_label = "mDPR" if tag == "dense" else "Hybrid"
    mdpr_color = "tab:orange" if tag == "dense" else "salmon"

    qrels = load_qrels(args.qrels_file)
    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)

    bm25_ranks = get_rank_list(qrels, bm25)
    mdpr_ranks = get_rank_list(qrels, mdpr)

    fig = plt.figure()
    ax = plt.gca()

    common_args = {"alpha": 0.4}
    # bins = [-10] + list(range(0, 110, 10))
    bins = list(range(0, 110, 10)) + [110]
    # bins = [-100] + list(range(0, 1100, 100))
    bm25_n, bm25_patches = plot_hist(bm25_ranks, bins=bins, ax=ax, direction="pos", label="BM25", color="tab:blue", **common_args)
    mdpr_n, mdpr_patches = plot_hist(mdpr_ranks, bins=bins, ax=ax, direction="pos", label=mdpr_label, color=mdpr_color, **common_args)

    # plt.xticks(bins, ["Not Found"] + bins[1:])
    plt.xticks(bins[:-1] + [bins[-1], bins[-1]+5], bins[:-1] + ["", "Not Found"])

    plt.grid(color="lightgray")
    plt.xlabel("Top-k documents")
    plt.ylabel("No. of Q with relevant document in top-k")
    lang = args.lang.capitalize()

    # plot the right axis
    ax_right = ax.twinx()
    x_s = [
        # (bins[i] + bins[i + 1]) / 2 for i in range(1, len(bins) - 1)
        (bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 2)
    ]
    common_args = {"alpha": 1}
    bm25_ratio_patches = plot_ratio(bm25_n[:-1], x_s, ax=ax_right, color="tab:blue", label="BM25 (normalized)", **common_args)
    mdpr_ratio_patches = plot_ratio(mdpr_n[:-1], x_s, ax=ax_right, color=mdpr_color, label=f"{mdpr_label} (normalized)", **common_args)
    ax_right.set_ylabel("Distribution of Q with relevant document in top-100")
    ax_right.set_ylim(0, 1)

    # combined all labels
    all_patches = bm25_patches.patches + mdpr_patches.patches + bm25_ratio_patches + mdpr_ratio_patches
    labels = [l.get_label() for l in all_patches]
    ax_right.legend(all_patches, labels, loc=9, ncol=2) # upper center

    plt.title(lang)
    plt.tight_layout()

    dir = f"{plot_dir}/rank-hist-pdf/{tag}"
    os.makedirs(dir, exist_ok=True)
    # plt.savefig(f"{dir}/rank-hist-{lang}-{tag}.png")
    plt.savefig(f"{dir}/rank-hist-{lang}-{tag}.pdf")


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
