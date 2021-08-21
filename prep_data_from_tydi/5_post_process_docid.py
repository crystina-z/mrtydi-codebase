"""
The purpose of this file:
1. f"{docid}-{rel_id}" to f"{docid}#{rel_id}"
2. json.dumps(info, ensure_ascii=False)
# 3. gzip collection
"""

import os
import sys
import pdb
import json
from argparse import ArgumentParser

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os.path.abspath(__file__)))
sys.path.append(PACKAGE_PATH)

from utils import lang_full2abbr, LANGS
from pyserini.analysis import Analyzer, get_lucene_analyzer


FILES = {
    "to_change": [
        'collection/docs.jsonl', 
        'qrels.txt', 
        'qrels.train.txt', 
        'qrels.dev.txt', 
        'qrels.test.txt', 
    ],
    "copy": [
        'pid2passage.tsv', # only contain the mapping between Wiki title and Wiki docid
        'topic.tsv', 
        'topic.train.tsv'
        'topic.dev.tsv', 
        'topic.test.tsv', 
        'folds.json', 
    ],
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mrtydi-dir", "-d", type=str, required=True)
    return parser.parse_args()


def update_id(docid):
    assert len(docid.split("-")) == 2
    return docid.replace("-", "#")


def change_doc_jsonl(ori_file, out_file, analyze_fn):
    n_empty, n_kept = 0, 0
    empty_ids_fn = open(os.path.splitext(ori_file)[0].replace("collection/", "") + ".empty-ids", "w")
    os.makedirs(os_dir(out_file), exist_ok=True)

    with open(ori_file) as f, open(out_file, "w") as fout: 
        for line in f:
            line = json.loads(line)
            tokens = analyze_fn(line["contents"])
            if len(tokens) == 0 or line["contents"].strip() in {"__NOEDITSECTION__", }: 
                n_empty += 1
                empty_ids_fn.write(json.dumps(line, ensure_ascii=False))
                continue

            # if len(tokens) == 1:
            #     print(tokens, line)

            n_kept += 1
            docid = update_id(line["id"])
            line["id"] = docid
            line = json.dumps(line, ensure_ascii=False)
            fout.write(line + "\n")

    print(f"{n_empty} empty docs were found in {ori_file}; {n_kept} docs were kept.")


def change_qrel(ori_file, out_file):
    os.makedirs(os_dir(out_file), exist_ok=True)
    with open(ori_file) as f, open(out_file, "w") as fout: 
        for line in f:
            line = line.strip().split()  # qid, Q0, docid, label\n
            line[2] = update_id(line[2])
            assert "#" in line[2]
            fout.write(" ".join(line) + "\n")
 

def main(args):
    mrtydi_dir = args.mrtydi_dir
    output_dir = os_join(mrtydi_dir, os.pardir, "dataset_processed_docid")

    for lang in LANGS:
        print(f"*** post-processing {lang} ***")
        cur_input_dir = os_join(mrtydi_dir, lang)
        cur_outp_dir = os_join(output_dir, lang)
        analyze_fn = Analyzer(get_lucene_analyzer(language=lang_full2abbr[lang])).analyze \
            if lang not in ["swahili", "telugu"] else lambda word: word.split()

        for fn in FILES["to_change"]:
            ori_file = os_join(cur_input_dir, fn)
            outp_file = os_join(cur_outp_dir, fn)
            if fn.startswith("qrels"):
                change_qrel(ori_file, outp_file)
            else:
                change_doc_jsonl(ori_file, outp_file, analyze_fn)


if __name__ == "__main__":
    args = get_args()
    main(args)
