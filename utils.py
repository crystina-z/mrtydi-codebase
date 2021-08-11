import os
import gzip
import json 
import subprocess
from collections import defaultdict, OrderedDict

import numpy as np


lang_full2abbr = OrderedDict({
    "thai": "th",
    "swahili": "sw",
    "telugu": "te",
    "finnish": "fi",
    "bengali": "bn",
    "russian": "ru",
    "japanese": "ja",
    "arabic": "ar",
    "indonesian": "id",
    "korean": "ko",
    "english": "en",
})
LANGS = lang_full2abbr


def index(lst, element):
    try:
        return lst.index(element)
    except:
        return len(lst) + 1


def str_list(lst):
    return list(map(str, lst)) 


def sort_qid_docid_value_dict(d):
    sorted_d = OrderedDict()
    try:
        qids = sorted(
            d.keys(), key=lambda k: int(k)
        )  # sort according to qid int value rather than string value
    except:
        qids = sorted(d.keys())

    for qid in qids:  # sort according to label/score, from large to small
        docs = sorted(d[qid].items(), key=lambda kv: kv[1], reverse=True)
        sorted_d[qid] = {k: v for k, v in docs}
    return sorted_d


def write_qrels(qrels_dict, outp_fn):
    os.makedirs(os.path.dirname(outp_fn), exist_ok=True)
    sorted_qrels = sort_qid_docid_value_dict(qrels_dict)
    with open(outp_fn, "w", encoding="utf-8") as f:
        for qid in sorted_qrels:
            for docid, value in sorted_qrels[qid].items():
                f.write(f"{qid}\tQ0\t{docid}\t{value}\n")


def write_to_topic_tsv(topic2id, outp_topic_tsv):
    with open(outp_topic_tsv, "w") as f:
        for topic, qid in topic2id.items():
            f.write(f"{qid}\t{topic}\n")


def load_topic_tsv(fn):
    id2topic = {}
    with open(fn) as f:
        for line in f:
            qid, topic = line.strip().split("\t")
            id2topic[qid] = topic
    return id2topic


def document_to_trectxt(docno, txt):
    s = f"<DOC>\n<DOCNO> {docno} </DOCNO>\n"
    s += f"<TEXT>\n{txt}\n</TEXT>\n</DOC>\n"
    return s


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


def load_collection_trec(coll_fn):
    """
    Return a iterator yielding doc id and contnet from trec-format collection file
    :param coll_fn: path to the trec-format collection file
    :return: a iterator yielding (docid, document content)
    """
    docid = ""
    f = gzip.open(coll_fn) if coll_fn.endswith(".gz") else open(coll_fn, "rb")

    def read_nextline():
        while True:
            try:
                line = f.readline()
                line = line.decode().strip()
                break
            except:
                print(f"invalid line:\t {line}")
        return line

    while True:
        line = read_nextline()
        if line == "":
            line = read_nextline()
            if line == "":
                break

        if line.startswith("<DOCNO>"):
            docid = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()

        if line == "<TEXT>":
            doc = read_nextline()
            while True:
                line = read_nextline()
                if line == "</TEXT>":
                    break
                doc += line

            assert docid != ""
            yield docid, doc.strip()
            docid = ""


def load_collection_jsonl(coll_fn):
    with open(coll_fn) as f:
        for line in f:
            line = json.loads(line)
            yield line["id"], line["contents"]


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


def byte_str(text):
  return text.encode("utf-8")


def byte_slice(text, start, end, errors="replace"):
  # Python 3 encodes text as character sequences, not byte sequences (like Python 2).
  return byte_str(text)[start:end].decode("utf-8", errors=errors)

