import os
import matplotlib.pyplot as plt
from matplotlib import colors as mat_colors

import pandas as pd


bar_width = 0.8
distance = 1
colors = ["tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple"]

scores = pd.read_csv("stats/scores.tsv", delimiter="\t", header=0)
n_rows = len(scores)

methods = scores["method"]
valid_methods = methods[methods != "BM25"]
n_methods = len(valid_methods)
method2color = dict(zip(valid_methods, colors))


i = 0
xs, langs = [], []
for lang in scores:
    if lang == "method":
        continue
    langs.append(lang)
    xs.append(i)

    lang_scores = scores[lang][methods != "BM25"]
    bm25 = scores[lang][methods == "BM25-tuned"].item()
    x_start = i - (bar_width * n_methods / 2)

    for j, (method, score) in enumerate(zip(valid_methods, lang_scores)):
        # color = 
        rel_score = score / bm25

        cur_x = x_start + bar_width * j
        edge_color = method2color[method]
        face_color = (*mat_colors.to_rgba(edge_color)[:3], 0.6)

        if i == 0:
            plt.bar(
                x=cur_x, width=bar_width, height=rel_score, color=face_color, edgecolor=edge_color, label=method)
        else:
            plt.bar(
                x=cur_x, width=bar_width, height=rel_score, color=face_color, edgecolor=edge_color)

    i += (bar_width * n_methods + distance)


file_dir = os.path.dirname(__file__) 
plot_dir = os.path.join(file_dir, "plots")

plt.legend()
plt.xticks(xs, langs)
plt.tight_layout()
plt.savefig(f"{plot_dir}/relative_score_bars.png")