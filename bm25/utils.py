import subprocess
from collections import defaultdict


def str_list(lst):
    return list(map(str, lst)) 


def load_qrels(fn):
    """
    Loading trec format query relevance file into a dictionary
    :param fn: qrel file path
    :return: dict, in format {qid: {docid: label, ...}, ...}
    """
    qrels = defaultdict(dict)
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, label = line.strip().split()
            qrels[qid][docid] = int(label)
    return qrels


def load_runs(fn):
    """
    Loading trec format runfile into a dictionary
    :param fn: runfile path
    :return: dict, in format {qid: {docid: score, ...}, ...}
    """
    runs = defaultdict(dict)
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _, score, _ = line.strip().split()
            runs[qid][docid] = float(score)
    return runs


def run_command(cmd):
    rtn = subprocess.run(cmd) 
    if rtn.returncode != 0:
        print(" ".join(cmd) + "\tCommand Failed") 
        raise ValueError("returncode of cmd is not 0")


def aggregate_score(qid2score):
    aggregated = defaultdict(list)
    for qid in qid2score:
        for name, value in qid2score[qid].items():
            aggregated[name].append(value)         

    for name in aggregated:
        aggregated[name] = np.mean(aggregated[name])

    return aggregated
