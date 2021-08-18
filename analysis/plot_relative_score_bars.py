import os
import matplotlib.pyplot as plt
from matplotlib import colors as mat_colors

import pandas as pd


figsize=[8, 6]
boxsize=[4.5, 4.5]
bar_width = 0.8
distance = 1
colors = ["tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple"]

# scores = pd.read_csv("stats/scores.tsv", delimiter="\t", header=0)
scores = pd.read_csv("stats/scores-top100.tsv", delimiter="\t", header=0)
# scores = pd.read_csv("stats/scores-em.tsv", delimiter="\t", header=0)
n_rows = len(scores)


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)

def filter_fn(df):
    return (df != "BM25") & (df != "BM25-tuned")


methods = scores["method"]
valid_methods = methods[filter_fn(methods)]
n_methods = len(valid_methods)
method2color = dict(zip(valid_methods, colors))


def barplot():
    i = 0
    plt.figure(figsize=figsize)
    xs, langs = [], []
    for lang in scores:
        if lang == "method":
            continue
        langs.append(lang)
        xs.append(i)

        # lang_scores = scores[lang][methods != "BM25"]
        bm25 = scores[lang][methods == "BM25-tuned"].item()
        lang_scores = scores[lang][filter_fn(methods)]

        x_start = i - (bar_width * (n_methods - 1) / 2)

        for j, (method, score) in enumerate(zip(valid_methods, lang_scores)):
            rel_score = round(score / bm25, 2)

            cur_x = x_start + bar_width * j
            edge_color = method2color[method]
            face_color = (*mat_colors.to_rgba(edge_color)[:3], 0.6)

            if i == 0:
                rects = plt.bar(
                    x=cur_x,
                    width=bar_width,
                    height=rel_score,
                    color=face_color,
                    edgecolor=edge_color,
                    label=method,
                )
            else:
                rects = plt.bar(
                    x=cur_x,
                    width=bar_width,
                    height=rel_score,
                    color=face_color,
                    edgecolor=edge_color,
                )
            plt.bar_label(rects, padding=1)

        i += bar_width * n_methods + distance

    plt.plot(
        [xs[0] - bar_width * n_methods, xs[-1] + bar_width],
        [1, 1],
        label="BM25 (tuned)",
        linestyle="--",
        color="tab:grey",
    )

    file_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(file_dir, "plots")

    plt.legend()
    plt.xticks(xs, langs)
    plt.ylabel("MRR normalized to BM25 (tuned)")
    set_size(*boxsize)
    # plt.tight_layout()
    plt.savefig(f"{plot_dir}/relative_score_bars-top100.png")


def scatter():
    i = 0
    xs, langs = [], []

    # lang 2 all score
    lang2all_scores = {
        lang: {
            "bm25": scores[lang][methods == "BM25-tuned"].item(),
            "mDPR": scores[lang][methods == "mDPR"].item(),
            "hybrid": scores[lang][methods == "hybrid"].item(),
        }
        for lang in scores
        if lang != "method"
    }
    lang2rel_scores = {
        lang: {method: score / scores["bm25"] for method, score in scores.items()}
        for lang, scores in lang2all_scores.items()
    }
    lang_rel_scores = sorted(
        lang2rel_scores.items(), key=lambda kv: (kv[1]["mDPR"], kv[1]["hybrid"])
    )
    for i, (lang, rel_scores) in enumerate(lang_rel_scores):
        plt.scatter(rel_scores["mDPR"], rel_scores["hybrid"], s=5, color="tab:blue")
        plt.text(s=lang, x=rel_scores["mDPR"] * 1.015, y=rel_scores["hybrid"]) 

    file_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(file_dir, "plots")

    # plt.legend()
    plt.xlabel("MRR of mDPR normalized to BM25 (tuned)")
    plt.ylabel("MRR of sparse-dense hybrid normalized to BM25 (tuned)")
    # plt.tight_layout()
    set_size(*boxsize)
    plt.savefig(f"{plot_dir}/relative_score_scatter-top100.png")


if __name__ == "__main__":
    barplot()
    # scatter()
