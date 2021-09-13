import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_dir))
from utils import LANGS

skips = ["En", "Ru"]

outp_dir = "plots/exclusive-lang" 
os.makedirs(outp_dir, exist_ok=True)

df = pd.read_csv("stats/exclusive-scores.tsv", delimiter="\t")
exclude_filter = lambda name: 'no' in name
exclude_df = df[[exclude_filter(name) for name in df['method']]]
# mDPR_filter = lambda name: "mDPR" in name
# mDPR_df = df[[mDPR_filter(name) for name in df['method']]]

plt.figure(figsize=(15, 10))
for i, col_name in enumerate(df):
    if col_name == "method":
        continue

    print(col_name)

    # outp_fig_path = os.path.join(outp_dir, f"{col_name}.png")

    col = exclude_df[col_name]
    xs = list(range(len(col)))
    # plt.figure()
    plt.subplot(3, 4, i)

    if col_name in skips:
        plt.ylim(0, 1)
        plt.title(col_name)
        continue

    zero_shot = df[df["method"] == "mDPR"][col_name]
    full = df[df["method"] == "mDPR-all-data"][col_name]
    sampled = df[df["method"] == "mDPR-sampled-data"][col_name]

    plt.scatter(xs, col, s=5)
    plt.plot([xs[0], xs[-1]], [zero_shot, zero_shot], label="zero shot", alpha=0.5)
    plt.plot([xs[0], xs[-1]], [full, full], label="ft (all data)", alpha=0.5)
    plt.plot([xs[0], xs[-1]], [sampled, sampled], label="ft (sampled data)", alpha=0.5)

    x_labels = ["no-" + LANGS[m.replace("no-", "")] for m in exclude_df["method"]]
    plt.xticks(xs, x_labels, rotation=55)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(color="grey", alpha=0.4, linestyle="--")
    plt.title(col_name)

    # plt.savefig(outp_fig_path)
    # plt.close()
 
plt.tight_layout()
plt.savefig(os.path.join(outp_dir, f"all.png"))
