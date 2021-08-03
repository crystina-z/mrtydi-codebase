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
        # ax = plt.axes(figure=fig, arg=(0,0,0,0))
        n, _, _ = ax.hist(x=data, bins=bins)
        del ax, fig
        # n, _, _ = plt.hist(x=data, bins=bins)
        # del fig
        return n
    else:
        # pdb.set_trace()
        ax.hist(x=data, bins=bins, **kwargs)


def plot_hist(data, bins, ax, direction="pos", **kwargs):
    assert direction in {"pos", "neg"}
    if direction == "pos":  # normal plots
        _hist(data, bins, ax=ax, **kwargs)
    elif direction == "neg":  # normal plots
        n = _hist(data, bins)
        # pdb.set_trace()
        assert len(n) == len(bins) - 1
        neg_n = [-value for value in n]
        _hist(data, bins, ax=ax, bottom=neg_n, **kwargs)  # plot the histogram at the neg axis


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

    qrels = load_qrels(args.qrels_file)
    bm25 = load_runs(args.bm25_runfile)
    mdpr = load_runs(args.mdpr_runfile)

    bm25_ranks = get_rank_list(qrels, bm25)
    mdpr_ranks = get_rank_list(qrels, mdpr)

    fig = plt.figure()
    ax = plt.gca()

    common_args = {"alpha": 0.5}
    # bins = [-1] + list(range(0, 60, 10)) + [100, 1000]
    # bins = [-1] + list(range(0, 60, 10)) + [100]
    bins = [-10] + list(range(0, 110, 10))
    plot_hist(bm25_ranks, bins=bins, ax=ax, direction="pos", label="BM25", color="tab:blue", **common_args)
    # plot_hist(bm25_ranks, bins=bins, ax=ax, direction="neg", color="tab:blue", **common_args)
    label = "mDPR" if tag == "dense" else "Hybrid"
    # plot_hist(mdpr_ranks, bins=bins, ax=ax, direction="neg", label=label, color="tab:orange", **common_args)
    plot_hist(mdpr_ranks, bins=bins, ax=ax, direction="pos", label=label, color="tab:orange", **common_args)

    plt.xticks(bins, ["Unfound"] + bins[1:])

    plt.legend()
    plt.grid(color="lightgray")
    plt.xlabel("Top-k documents")
    plt.ylabel("No. of Q with relevant document")
    lang = args.lang.capitalize()

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
