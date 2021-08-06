import os
import matplotlib.pyplot as plt
from matplotlib import colors as mat_colors

import pandas as pd


bar_width = 0.8
distance = 1
colors = ["tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple"]

scores = pd.read_csv("stats/scores.tsv", delimiter="\t", header=0)
n_rows = len(scores)


def filter_fn(df):
    return (df != "BM25") & (df != "BM25-tuned")


methods = scores["method"]
valid_methods = methods[filter_fn(methods)]
n_methods = len(valid_methods)
method2color = dict(zip(valid_methods, colors))


def barplot():
    i = 0
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
    plt.ylabel("Relative MRR comparing to tuned BM25")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/relative_score_bars.png")


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

    # sort according to relevant mdpr score
    lang_rel_scores = sorted(
        lang2rel_scores.items(), key=lambda kv: (kv[1]["mDPR"], kv[1]["hybrid"])
    )

    # plot
    for i, (lang, rel_scores) in enumerate(lang_rel_scores):
        for method in ["mDPR", "hybrid"]:
            common_kwargs = {
                "color": method2color[method],
                "s": 5,
            }
            if i == 0:
                plt.scatter(i, rel_scores[method], label=method, **common_kwargs)
            else:
                plt.scatter(i, rel_scores[method], **common_kwargs)
            plt.text(s=lang, x=i + 0.1, y=rel_scores[method])

    xticks, _ = plt.xticks()
    # print(xticks[0], xticks[-1])
    plt.plot(
        [-0.5, len(lang_rel_scores) - 0.5],
        [1, 1],
        color="tab:grey",
        label="BM25 (tuned)",
        linestyle="--",
    )

    file_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(file_dir, "plots")

    plt.legend()
    # plt.xticks(xs, langs)
    plt.ylabel("Relative MRR comparing to tuned BM25")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/relative_score_scatter.png")


if __name__ == "__main__":
    # barplot()
    scatter()
