import os
import matplotlib.pyplot as plt

import pandas as pd

# 2021 version
# https://meta.wikimedia.org/wiki/List_of_Wikimedia_projects_by_size
# n_training_data_rank = [
#     "En", # 4
#     "Ru", # 11
#     "Ja", # 17
#     "Ar", # 23
#     "Id", # 42
#     "Ko", # 45
#     "Fi", # 47
#     "Th", # 106
#     "Bn", # 120
#     "Te", # 147
#     "Sw", # 149 
# ]

# 2019 Jan version
# https://meta.wikimedia.org/w/index.php?title=List_of_Wikimedia_projects_by_size/Table&oldid=18751492
n_training_data_rank = [
    "En", # 4
    "Ru", # 12
    "Ja", # 18
    "Ar", # 31 
    "Id", # 43
    "Ko", # 45
    "Fi", # 46
    "Th", # 103
    "Te", # 135
    "Bn", # 141
    "Sw", # 154
]

scores = pd.read_csv("stats/scores.tsv", delimiter="\t", header=0)
import pdb

plt.figure(figsize=(8, 8))
i = 1

for _, row in scores.iterrows():
    method = row.iloc[0]
    if "BM25" in method:
        continue

    scores = row.iloc[1:].sort_values(ascending=False).tolist()  # high -> low
    score_ranks = row.iloc[1:].sort_values(ascending=False).index.tolist()  # high -> low

    plt.subplot(2, 2, i)
    xs = list(range(len(n_training_data_rank)))
    ys = [score_ranks.index(lang) for lang in n_training_data_rank]
    plt.scatter(xs, ys, s=5)
    for x, y, lang in zip(xs, ys, n_training_data_rank):
        plt.text(x=x+0.3, y=y-0.1, s=lang, fontsize='x-small')

    plt.plot([0, 10], [0, 10], color="tab:orange", linestyle="--")
    plt.xlabel("Rank by pretraining data size")
    plt.ylabel(f"Rank by {method}")
    plt.title(method + " (Rank)")

    plt.subplot(2, 2, i + 1)
    xs = [10 - x for x in xs]
    ys = [scores[score_ranks.index(lang)] for lang in n_training_data_rank]
    plt.scatter(xs, ys, s=5)
    for x, y, lang in zip(xs, ys, n_training_data_rank):
        plt.text(x=x+0.3, y=y, s=lang, fontsize='x-small')

    plt.xlabel("Pretraining data set size")
    plt.ylabel(f"Score of {method}")
    plt.ylim(bottom=0, top=1)
    plt.title(method + " (Score)")

    i += 2


file_dir = os.path.dirname(__file__) 
plot_dir = os.path.join(file_dir, "plots")

plt.tight_layout()
plt.savefig(f"{plot_dir}/n-training-data-vs-score.png")