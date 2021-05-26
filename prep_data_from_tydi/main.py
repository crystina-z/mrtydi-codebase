import os
import json
import subprocess

from pprint import pprint 
from collections import defaultdict
from argparse import ArgumentParser

from tqdm import tqdm
from lxml.html import fromstring
from nirtools.ir import write_qrels
from capreolus.utils.trec import topic_to_trectxt, document_to_trectxt


LANGS = ["bn", "ar"]
lang2lang = {
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
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--tydi_dir", type=str, required=True, 
        help="The directory that contain the train and dev jsonl of TyDi dataset.")
    parser.add_argument(
        "--wiki_dir", type=str, required=True, 
        help="The directory that contain the unziped wikipedia file of all languages")
    parser.add_argument(
        "--output_dir", type=str, required=True, 
        help="The output directory that stores the parsed benchmark and collections")

    return parser.parse_args()


def jsonl_loader(jsonl_path):
    with open(jsonl_path) as f:
        for line in f:
            line = json.loads(line)
            question = line["question_text"]
            doc_title, doc_url = line["document_title"], line["document_url"]
            lang = line["language"]

            # doc = line["document_plaintext"]
            # todo: what does 'annotation' and 'passage_answer_candidates' mean respectively?
            yield lang, question, (doc_url, doc_title)


def print_lang_stats(jsonl_path, print_out=False):
    lang2num = defaultdict(int)
    lang2query_list = defaultdict(list)
    for lang, question, _ in jsonl_loader(jsonl_path):
        lang2num[lang] += 1
        lang2query_list[lang].append(question)
    
    if print_out:
        print(f"{'Language':>10}{'Total No.':>10}{'Uniq. No.':>10}{'Dup. No.':>10}")
        for lang in sorted(lang2num):
            num = lang2num[lang]
            queries, uniq_queries = lang2query_list[lang], set(lang2query_list[lang])
            uniq_num = len([q for q in uniq_queries if queries.count(q) == 1])
            dup_num = len([q for q in uniq_queries if queries.count(q) > 1])

            print(f"{lang:>10}{num:>10}{uniq_num:>10}{dup_num:>10}")

    return lang2num


def prepare_doc2id_from_wiki(wiki_json, output_dir, prepare_trec_coll=True):
    wiki_dir, name = os.path.dirname(wiki_json), os.path.basename(wiki_json).split("-")[0] 
    doc2id_file = os.path.join(output_dir, f"{name}.doc2id.tsv")
    if prepare_trec_coll:
        coll_file = os.path.join(output_dir, f"{name}.trec.collection.txt")
        coll_file_f = open(coll_file, "w") 

    print(f"Preparing collection from {name}...")

    with open(wiki_json) as f, open(doc2id_file, "w") as fout:
        for line in f: 
            line = json.loads(line) 
            docid, url, title = line["id"], line["url"], line["title"]
            fout.write(f"{url}\t{title}\t{docid}\n")

            if prepare_trec_coll:
                doc = line["text"]
                try:
                    doc = fromstring(doc).text_content()
                except Exception as e:
                    print(docid, len(doc))
                doc.lstrip(title)
                coll_file_f.write(document_to_trectxt(docid, doc)) 

    if prepare_trec_coll:
        coll_file_f.close()

    return doc2id_file


def load_doc2id(doc2id_tsv):
    """ Load {url: {title: docid}} from the tsv file of format `url title  docid` per line """
    url2title2docid = {}
    with open(doc2id_tsv) as f:
        for line in f:
            url, title, docid = line.strip().split("\t") 
            if title in url2title2docid:
                import pdb
                pdb.set_trace()
                raise Value(f"Duplicate key pairs: {url} {title}")

            url2title2docid[title] = docid
    return url2title2docid


def write_to_topic_tsv(topic2id, outp_topic_tsv):
    with open(outp_topic_tsv, "w") as f:
        for topic, qid in topic2id.items():
            f.write(f"{qid}\t{topic}\n")


def prepare_benchmarks_from_tydi(tydi_dir, lang2doc2id, output_dir):
    lang2topics, lang2qrels, lang2folds = {}, {}, {} 
    # init
    for lang in LANGS:
        lang2topics[lang] = {} 
        lang2qrels[lang] = defaultdict(dict) 
        lang2folds[lang] = {"train": set(), "dev": set()}

    sets = ["train", "dev"]
    for set_name in sets: 
        fn = f"tydiqa-v1.0-{set_name}.jsonl"

        jsonl_path = os.path.join(tydi_dir, fn) 
        for lang, question, (doc_url, doc_title) in jsonl_loader(jsonl_path):
            lang = lang2lang[lang]
            if lang not in LANGS:
                continue

            # add to topic
            if question not in lang2topics[lang]:
                lang2topics[lang][question] = len(lang2topics[lang])

            qid = lang2topics[lang][question]
            docid = lang2doc2id[lang].get(doc_title)
            if docid is None:
                print(f"Warning: could not find doc id for {doc_title}")
                continue

            # todo: is this true? all documents appearing here are relevant? 
            lang2qrels[lang][qid][docid] = 1  # add to qrels 
            lang2folds[lang][set_name].add(qid)  # add to folds

    for lang in LANGS:
        lang_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        topic_fn, qrel_fn, fold_fn = \
            os.path.join(lang_dir, "topic.tsv"), os.path.join(lang_dir, "qrels.txt"), os.path.join(lang_dir, "folds.json")

        if all([os.path.exists(fn) for fn in [topic_fn, qrel_fn, fold_fn]]):
            print(f"all files found for {lang}. skip")
            continue

        print(f"Dumping benchmark files of {lang}...")

        topics, qrels, folds = lang2topics[lang], lang2qrels[lang], lang2folds[lang]
        folds = {k: list(v) for k, v in folds.items()}
        write_to_topic_tsv(topics, topic_fn)
        write_qrels(qrels, qrel_fn)
        json.dump(folds, open(fold_fn, "w"))


def main(args):
    """ 
    Group the train and dev entries by language. 
    For each language, prepare: 
        (1) topic.tsv; (2) qrels.txt; (3) folds.json (4) docid_2_url_and_title.json
    """
    wiki_dir, tydi_dir = args.wiki_dir, args.tydi_dir
    output_dir = args.output_dir

    doc2id_output_dir = os.path.join(output_dir, "doc2id") 
    os.makedirs(doc2id_output_dir, exist_ok=True)

    lang2doc2id = {}
    for lang in LANGS:
        wiki_fn = os.path.join(wiki_dir, f"{lang}wiki.20190201.json")
        doc2id_fn = prepare_doc2id_from_wiki(wiki_json=wiki_fn, output_dir=doc2id_output_dir)
        lang2doc2id[lang] = load_doc2id(doc2id_fn)
    
    prepare_benchmarks_from_tydi(tydi_dir=tydi_dir, lang2doc2id=lang2doc2id, output_dir=output_dir)


def print_stats_main(args):
    tydi_dir = args.tydi_dir
    print(">>> dev set <<")
    print_lang_stats(os.path.join(tydi_dir, f"tydiqa-v1.0-dev.jsonl"), print_out=True)

    print(">>> train set <<")
    print_lang_stats(os.path.join(tydi_dir, f"tydiqa-v1.0-train.jsonl"), print_out=True)


if __name__ == "__main__":
    args = get_args()
    main(args)
    # print_stats_main(args)
