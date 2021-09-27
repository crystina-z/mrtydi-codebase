import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import colors as mat_colors

figsize=[8, 6]
boxsize=[4.5, 4.5]
bar_width = 0.8
distance = 1
colors = ["tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple"]

# scores = pd.read_csv("stats/scores.tsv", delimiter="\t", header=0)
scores = pd.read_csv("stats/scores-top100.tsv", delimiter="\t", header=0)
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
            labels = ["%.2f" % round(rel_score, 2)]
            plt.bar_label(rects, labels=labels, padding=1)

        i += bar_width * n_methods + distance

    plt.plot(
        [xs[0] - bar_width * n_methods, xs[-1] + bar_width * 2],
        [1, 1],
        label="BM25 (tuned)",
        linestyle="--",
        color="tab:grey",
    )

    file_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(file_dir, "plots")

    plt.legend()
    plt.xticks(xs, langs)
    plt.ylabel("MRR@100 normalized to BM25 (tuned)")
    set_size(*boxsize)
    # plt.tight_layout()
    plt.savefig(f"{plot_dir}/relative_score_bars-top100.pdf")


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

    x_s, y_s = [], []
    for i, (lang, rel_scores) in enumerate(lang_rel_scores):
        plt.scatter(rel_scores["mDPR"], rel_scores["hybrid"], s=5, color="tab:blue")
        plt.text(s=lang, x=rel_scores["mDPR"] * 1.015, y=rel_scores["hybrid"]) 
        x_s.append(rel_scores["mDPR"])
        y_s.append(rel_scores["hybrid"])
        # print(rel_scores)
    
    # linear regression
    x_s, y_s = np.array(x_s), np.array(y_s)
    m, b = np.polyfit(x_s, y_s, 1)

    # plot linear regression
    # x_s_to_plot = np.concatenate(([0.0], x_s, [x_s.max() + 0.1]))
    x_s_to_plot = np.concatenate(([0.05], x_s, [x_s.max() + 0.05]))
    lr_y_s_to_plot = m * x_s_to_plot + b
    plt.plot(
        x_s_to_plot, lr_y_s_to_plot, 
        color="tab:grey",
        linestyle="--", 
        alpha=0.5,
        # label="Linear Regression"
    )

    # R^2
    # from sklearn.metrics import r2_score
    # R_square = r2_score(x_s, y_s)
    y_mean = y_s.mean()
    lr_y_s = m * x_s + b  # pred y using linear regression
    diff_lr = ((lr_y_s - y_s) ** 2).sum()
    diff_mean = ((y_mean - y_s) ** 2).sum()
    R_square = (diff_mean - diff_lr) / diff_mean
    # print("R^2: ", R_square)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(
        s=f"$R^2$ = %.2f" % R_square, 
        x=0.9, y=0.9,
        bbox=props, 
    ) 

    file_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(file_dir, "plots")

    x_range = np.arange(0, 1.2, 0.2)
    y_range = np.arange(1, 2.1, 0.2)
    x_range_str = list(map(lambda f: "%.1f" % f, x_range))
    y_range_str = list(map(lambda f: "%.1f" % f, y_range))
    plt.xticks(x_range, x_range_str)
    plt.yticks(y_range, y_range_str)
    plt.xlim(x_range[0] - 0.05, x_range[-1] + 0.15)
    plt.ylim(y_range[0] - 0.15, y_range[-1] + 0.05)

    plt.grid(linestyle=":", alpha=0.4)
    plt.xlabel("MRR@100 of mDPR normalized to BM25 (tuned)")
    plt.ylabel("MRR@100 of sparse-dense hybrid normalized to BM25 (tuned)")
    # plt.tight_layout()
    set_size(*boxsize)
    plt.savefig(f"{plot_dir}/relative_score_scatter-top100.pdf")


if __name__ == "__main__":
    barplot()
    # scatter()
