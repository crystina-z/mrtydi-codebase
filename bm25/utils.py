from collections import defaultdict


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
