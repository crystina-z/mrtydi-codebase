import os
import sys
import json
import gzip
from glob import glob

from tqdm import tqdm
from nirtools.ir import load_topic_tsv, load_runs, load_qrels


os_dir = os.path.dirname
os_join = os.path.join
TOP_K = 30

file_dir = os_dir(os.path.abspath(__file__)) 
sys.path.append(os_dir(file_dir))
from utils import find_file


def get_file_line_number(fn):
    return int(os.popen(f"wc -l {fn}").readline().split()[0])


def convert_doc(ori_doc_dict):
    """
    Input format: {"id": ..., "contents": ...}
    Output format: {"docid": ..., "title": ..., "text": ...}
    """
    docid, contents = ori_doc_dict["id"], ori_doc_dict["contents"]
    title, document = contents.split("\n\n") if "\n\n" in contents else ("", contents)
    return {"docid": docid, "title": title, "text": document}


def convert_corpus(corpus_json_fn, output_json_fn):
    """
    Input format: {"id": ..., "contents": ...}
    Output format: {"docid": ..., "title": ..., "text": ...}
    """
    num_lines = get_file_line_number(corpus_json_fn)
    open_handler = gzip.open if corpus_json_fn.endswith(".gz") else open
    with open_handler(corpus_json_fn) as f, open(output_json_fn, "w") as fout:
        for line in tqdm(f, total=num_lines, desc="Parsing Corpus"):
            line = json.loads(line)
            # docid, contents = line["id"], line["contents"]
            # title, document = contents.split("\n\n") if "\n\n" in contents else ("", contents)
            converted_doc_dict = convert_doc(line)
            fout.write(json.dumps(converted_doc_dict, ensure_ascii=False) + "\n")


def convert_train_queries(topic_tsv, corpus_json_fn, qrel_fn, runfile, output_json_fn):
    """
    Input format: "qid\tquery\n"
    Output format: {"query_id": qid, "query": query}
    """
    runs = load_runs(runfile)
    qrels = load_qrels(qrel_fn)

    open_handler = gzip.open if corpus_json_fn.endswith(".gz") else open
    docs = [convert_doc(json.loads(line)) for line in open_handler(corpus_json_fn)]
    id2docs = {doc["docid"]: doc for doc in docs}

    with open(output_json_fn, "w") as fout:
        qid2query = [(qid, query) for qid, query in load_topic_tsv(topic_tsv)] 
        for qid, query in tqdm(qid2query, desc="Parsing Training set"): 
            doc2scores = runs.get(qid, {})
            docids = {docid for docid, _ in sorted(doc2scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]}

            if len(docids) == 0:
                continue
            assert qid in qrels, f"Cannot find {qid} in qrels."
 
            positive_docids = qrels[qid]
            negative_docids = [docid for docid in docids if docid not in positive_docids]

            fout.write(json.dumps({
                "query_id": qid, 
                "query": query,
                "positive_passages": [id2docs[docid] for docid in positive_docids],
                "negative_passages": [id2docs[docid] for docid in negative_docids],
            }, ensure_ascii=False) + "\n")


def convert_dev_test_queries(topic_tsv, output_json_fn):
    """
    Input format: "qid\tquery\n"
    Output format: {"query_id": qid, "query": query}
    """
    with open(output_json_fn, "w") as fout:
        for qid, query in load_topic_tsv(topic_tsv):
            fout.write(json.dumps({"query_id": qid, "query": query}, ensure_ascii=False) + "\n")
 

# helpers
def get_collection_json(coll_dir):
    corpus_json_fn = find_file(f"{coll_dir}/docs.jsonl*")
    return os_join(coll_dir, corpus_json_fn)


def get_bm25_train_file(bm25_dir):
    bm25_fn = find_file(f"{bm25_dir}/bm25.train.k1=*.b=*")
    return os_join(bm25_dir, bm25_fn)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 7_convert_to_hgf_format.py $open_retrieval_dir $lang")
        exit(-1)

    version = "v1.1"

    mrtydi_dataset_dir = sys.argv[1]
    lang = sys.argv[2]
    bm25_dir = os_join(os_dir(mrtydi_dataset_dir), "bm25-runfiles")
    output_dir = os_join(os_dir(mrtydi_dataset_dir), "hgf-format-dataset")

    lang_dir = os_join(mrtydi_dataset_dir, f"mrtydi-{version}-{lang}")
    lang_bm25_dir = os_join(bm25_dir, lang)
    lang_outp_dir = os_join(output_dir, f"mrtydi-{version}-{lang}")
    os.makedirs(lang_outp_dir, exist_ok=True)

    # prepare file path
    corpus_json_dir = os_join(lang_dir, "collection")
    corpus_json_fn = get_collection_json(corpus_json_dir)

    convert_corpus(corpus_json_fn, os_join(lang_outp_dir, "corpus.jsonl"))
    convert_train_queries(
        topic_tsv = os_join(lang_dir, "topic.train.tsv"),
        corpus_json_fn = os_join(lang_dir, corpus_json_fn),
        qrel_fn = os_join(lang_dir, "qrels.train.txt"),
        runfile = get_bm25_train_file(lang_bm25_dir),
        output_json_fn = os_join(lang_outp_dir, "train.jsonl")
    )
    for set_name in ["dev", "test"]:
        convert_dev_test_queries(
            topic_tsv = os_join(lang_dir, f"topic.{set_name}.tsv"),
            output_json_fn = os_join(lang_outp_dir, f"{set_name}.jsonl")
        )
