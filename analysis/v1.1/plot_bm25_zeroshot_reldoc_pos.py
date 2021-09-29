import os
import sys
from glob import glob
import matplotlib.pyplot as plt

import numpy as np

file_dir = os.path.dirname(__file__) 
sys.path.append(os.path.dirname(file_dir))
from utils import index, load_runs, load_qrels, find_files, LANGS



def sort_docid(docid2score):
    sorted_docid2score = sorted(docid2score.items(), key=lambda kv: kv[1], reverse=True)
    return [docid for docid, score in sorted_docid2score]


def plot_rank(sorted_ranks, color, label, **kwargs):
    """
    Assume the (document) ranks is sorted by a certain criteiria 
    """
    for i, ranks in enumerate(sorted_ranks):
        ranks = [min(ranks)]
        xs = [i for _ in range(len(ranks))]
        ys = ranks
        plt.scatter(xs, ys, color=color, alpha=0.5, label=label, **kwargs)


def main():
    plot_dir = f"{file_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    for lang in LANGS:
        if lang in ["swahili", "arabic"]:
            continue

        print(lang)
        qrel_file = f"data/{lang}/qrels.test.txt"
        bm25_runfile = find_file(f"runfiles/bm25/{lang}/bm25.test.k1=*.b=*")
        dpr_runfile = f"runfiles/DPR/run.dense.test.{lang}.trec"
        hyper_runfile = f"runfiles/DPR/run.hybrid.test.{lang}.trec"

        qrels = load_qrels(qrel_file)
        bm25 = load_runs(bm25_runfile)
        dpr = load_runs(dpr_runfile)
        hybrid = load_runs(hyper_runfile)

        bm25_ranks = {
            qid: [index(sort_docid(docid2score), rel_docid) for rel_docid in qrels[qid]] for qid, docid2score in bm25.items()
        }
        dpr_ranks = {
            qid: [index(sort_docid(docid2score), rel_docid) for rel_docid in qrels[qid]] for qid, docid2score in dpr.items()
        }
        hybrid_ranks = {
            qid: [index(sort_docid(docid2score), rel_docid) for rel_docid in qrels[qid]] for qid, docid2score in hybrid.items()
        }

        # sorted_qids_bm25_ranks = sorted(bm25_ranks.items(), key=lambda kv: np.mean(kv[1]))
        # sorted_qids_bm25_ranks = sorted(bm25_ranks.items(), key=lambda kv: min(kv[1]))

        all_qids = set(bm25_ranks) | set(dpr_ranks) | set(hybrid_ranks)
        sorted_qids = [qid for qid in sorted(all_qids, key=lambda id: list(map(min, (
            bm25_ranks.get(id, [1001]), hybrid_ranks.get(id, [1001]), dpr_ranks.get(id, [1001])
            # hybrid_ranks.get(id, [1001]), bm25_ranks.get(id, [1001]), dpr_ranks.get(id, [1001])
        ))))]

        sorted_bm25_ranks = [bm25_ranks.get(qid, [1001]) for qid in sorted_qids]
        sorted_dpr_ranks = [dpr_ranks.get(qid, [1001]) for qid in sorted_qids]
        sorted_hybrid_ranks = [hybrid_ranks.get(qid, [1001]) for qid in sorted_qids]
        # import pdb
        # pdb.set_trace()

        fig = plt.figure()
        plot_rank(sorted_bm25_ranks, color="tab:orange", label="bm25", s=3)
        plot_rank(sorted_dpr_ranks, color="tab:green", label="DPR", s=3)
        plot_rank(sorted_hybrid_ranks, color="tab:grey", label="Hybrid", marker="+", s=6)

        plt.ylim(top=20, bottom=0)
        plt.savefig(f"{plot_dir}/bm25_dpr_rank.{lang}.triplet.png")
        # fig.close()


if __name__ == "__main__":
    main()
