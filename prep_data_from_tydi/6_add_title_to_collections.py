import os
import sys
import json
import argparse

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os.path.abspath(__file__)))
sys.path.append(PACKAGE_PATH)

from utils import load_collection_jsonl, load_id2title, LANGS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrtydi-v1-dir", "-d", required=True, type=str)
    return parser.parse_args()


def main(args):
    mrtydi_v1_dir = args.mrtydi_v1_dir
    for lang in LANGS:
        print(f"Processing {lang}", end="\t")
        lang_dir = os_join(mrtydi_v1_dir, f"mrtydi-v1.0-{lang}")

        docid2title_fn = os_join(lang_dir, "pid2passage.tsv")
        docid2title = {pid: title for pid, title in load_id2title(docid2title_fn)}

        output_collection_dir = os_join(lang_dir, "collection+title")
        output_collection_jsonl = os_join(output_collection_dir, "docs.jsonl")
        os.makedirs(output_collection_dir, exist_ok=True)

        n_total, n_unfound_title = 0, 0
        input_collection_jsonl = os_join(lang_dir, "collection", "docs.jsonl.gz")
        with open(output_collection_jsonl, "w") as f:
            for pid, passage in load_collection_jsonl(input_collection_jsonl):
                n_total += 1
                docid, segid = pid.split("#")
                try:
                    title = docid2title[docid]
                except KeyError:
                    n_unfound_title += 1
                    title = ""

                f.write(
                    json.dumps({"id": pid, "contents": f"{title}. {passage}"}, ensure_ascii=False) + "\n"
                )
        print(f"finished, {n_unfound_title} / {n_total} passage does not have found title.")
 

if __name__ == "__main__":
    args = get_args()
    main(args)