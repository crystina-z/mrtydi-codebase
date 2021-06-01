import os
from collections import defaultdict, OrderedDict


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


def document_to_trectxt(docno, txt):
    s = f"<DOC>\n<DOCNO> {docno} </DOCNO>\n"
    s += f"<TEXT>\n{txt}\n</TEXT>\n</DOC>\n"
    return s

