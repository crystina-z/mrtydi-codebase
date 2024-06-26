import os
import sys
import json
import random

from argparse import ArgumentParser
from collections import defaultdict, OrderedDict

from lxml.html import fromstring

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os.path.abspath(__file__)))
sys.path.append(PACKAGE_PATH)

from utils import lang_full2abbr, LANGS
from utils import write_to_topic_tsv, write_qrels, document_to_trectxt, byte_str, byte_slice

random.seed(123)
TRAIN_RATIO = 0.8  # from the train set, we keep $TRAIN_RATIO qids in train set, and the rest in dev set


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--tydi_dir", type=str, required=True, 
        help="The directory containing the language-grouped TyDi train and dev jsonl files.")
    parser.add_argument(
        "--wiki_dir", type=str, required=True, 
        help="The directory containing the WikIR-extracted .json wikipedia files of each languages")
    parser.add_argument(
        "--output_dir", type=str, required=True, 
        help="The output directory that we will output the parsed benchmark and collection files")

    return parser.parse_args()


def jsonl_loader(jsonl_path, expected_lang):
    """ 
    Load the jsonl files with ignoring the "yes_no_anwser" and "minimal_answer" fields,
    which are related to the "Minimal Answer Span Task" only. 
    """ 
    with open(jsonl_path) as f:
        for line in f:
            line = json.loads(line)
            question = line["question_text"]
            assert expected_lang == line["language"], f"expect {expected_lang} but got {line['language']}"

            doc, doc_title = line["document_plaintext"], line["document_title"]
            annotations, passage_answer_candidates = line["annotations"], line["passage_answer_candidates"]

            passages = [
                " ".join(
                    # doc[span["plaintext_start_byte"]:span["plaintext_end_byte"] + 1].replace("\n", " ").split()
                    byte_slice(doc, span["plaintext_start_byte"], span["plaintext_end_byte"]).replace("\n", " ").split()
                )
                for span in passage_answer_candidates
            ]
            empty_psg = [p for p in passages if p.strip() == ""]

            rel_indexes = [
                a["passage_answer"]["candidate_index"] for a in annotations if a["passage_answer"]["candidate_index"] != -1]
            rel_indexes = list({
                index for index in rel_indexes if passages[index] != ""})

            try:
                passages = passages[:passages.index("")]
            except ValueError: 
                pass 
            assert(all([p.strip() != "" for p in passages]))
            assert(len([p for p in passages if p.strip() == ""])) == 0


            # Warning: passage cannot be filter out here (as below)! 
            # since we will need to use rel_indexes to identify the relevant passage later.
            # >> passages = [p for p in passages if p != ""]
            yield question, doc_title, passages, rel_indexes


def segment_wiki_doc(doc):
    # currently we identify the passages via \n\n
    # todo: this code is a rough simulation of how TyDi does it, 
    # pending to change - https://github.com/google-research-datasets/tydiqa/issues/11 
    doc = doc.replace("\n\n\n", "") # this \n\n\n seems to indicate a removed table etc.
    passages = [p for p in doc.split("\n\n") if p.strip() != ""]
    assert(all([p.strip() != "" for p in passages]))
    return passages


def load_psg_dict_from_wiki_json(wiki_json): 
    title2_id_psgs = {} 
    with open(wiki_json) as f:
        for line in f: 
            line = json.loads(line) 
            docid, url, title, doc = line["id"], line["url"], line["title"], line["text"]

            assert title not in title2_id_psgs, f"Got duplicate Wikipedia article, {title}" 
            try:
                doc = fromstring(doc).text_content()
            except Exception as e:  # use unprocessed Wiki articles if extracting fails 
                print(docid, len(doc))

            doc = doc.lstrip(title).strip()   # Wiki articles start with title
            passages = segment_wiki_doc(doc)
            if len(passages) == 0:
                continue

            if title == passages[0]:
                print(f"Warning, {docid} with {url} still have article title as the first paragraph")
            title2_id_psgs[title] = (docid, passages) 

    return title2_id_psgs 


def prepare_dataset_from_tydi(lang, tydi_dir, wiki_psg_dict, output_dir):
    """
    :params tydi_dir: the directory that contains the train and dev jsonl files of a single language  
    :params output_dir: the language-specific directory that we will output:
        (1) collection file
        (2) topic file
        (3) qrel file
        (4) folds file
        (5) id2passage mapping file 
    """

    topics = {} 
    passage2id = {} 
    qrels = defaultdict(dict) 
    folds = {"train": set(), "dev": set()}

    sets = ["train", "dev"]
    for set_name in sets: 
        fn = f"tydiqa-v1.0-{set_name}.{lang}.jsonl"
        jsonl_path = os_join(tydi_dir, fn) 

        n_unfound_doc = 0
        for question, doc_title, passages, rel_indexes in jsonl_loader(jsonl_path, expected_lang=lang):
            # add to topic
            if question not in topics:
                topics[question] = str(len(topics))

            qid = topics[question]

            docid, wiki_passages = wiki_psg_dict.get(doc_title, (None, None))  # todo: verify if wiki_doc == doc

            if docid is None:
                n_unfound_doc += 1
                continue 

            # add to folds and qrels
            folds[set_name].add(qid)
            for rel_id in rel_indexes:
                passage_id = f"{docid}#{rel_id}"
                qrels[qid][passage_id] = 1 

            # overwrite our own wiki psg with the official ones 
            wiki_psg_dict[doc_title] = (docid, passages)

        print(f"Number of Unfound Wikipedia doc in {set_name}: {n_unfound_doc}")


    # post-process folds
    n_dup = len(folds["train"] & folds["dev"])
    folds["dev"] = folds["dev"] - folds["train"]
    folds = {k: list(v) for k, v in folds.items()}

    train_set = [qid for qid in folds["train"] if random.random() < TRAIN_RATIO]
    dev_set = [qid for qid in folds["train"] if qid not in train_set]
    test_set = folds["dev"]
    folds = {"train": train_set, "dev": dev_set, "test": test_set}
    print(f"Removed {n_dup} duplicate qids between train and dev set")

    # write to files
    def all_fn_exists(fns):
        return all([os.path.exists(fn) for fn in fns]) 

    lang_dir = os_join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    topic_fn, qrel_fn, fold_fn = \
        os_join(lang_dir, "topic.tsv"), os_join(lang_dir, "qrels.txt"), os_join(lang_dir, "folds.json")
    coll_fn, id2passage_fn = \
        os_join(lang_dir, "collection", "docs.jsonl"), os_join(lang_dir, "pid2passage.tsv")
    os.makedirs(os_dir(coll_fn), exist_ok=True)

    if all_fn_exists([topic_fn, qrel_fn, fold_fn]): 
        print(f"all benchmark files found for {lang}. skip")
    else:
        print(f"Dumping benchmark files of {lang}...")
        write_to_topic_tsv(topics, topic_fn)
        write_qrels(qrels, qrel_fn)
        json.dump(folds, open(fold_fn, "w"))

    if all_fn_exists([coll_fn, id2passage_fn]):
        print(f"all collection files found for {lang}. skip")
    else:
        print(f"Dumping collection files of {lang}...")
        wiki_psg_sorted = sorted(wiki_psg_dict.items(), key=lambda kv: int(kv[1][0]))  # sort according to docid

        with open(id2passage_fn, "w") as id2psg_f, open(coll_fn, "w") as coll_f: 
            for title, (docid, passages) in wiki_psg_sorted:
                id2psg_f.write(f"{docid}\t{title}\n")
                for i, passage in enumerate(passages):
                    passage_id = f"{docid}-{i}"
                    assert passage.strip() != "", f"Got empty passage for document {docid}, passage {i}, which have {len(passages)} doc in total"
                    # coll_f.write(document_to_trectxt(passage_id, passage))
                    coll_f.write(json.dumps({"id": passage_id, "contents": passage}) + "\n")


def main(args):
    """ 
    Group the train and dev entries by language. 
    For each language, prepare: 
        (1) topic.tsv; (2) qrels.txt; (3) folds.json (4) docid_2_url_and_title.json
    Note that in the folds.json; the train and dev set are prepared from the TyDi train set, and the test set is from the TyDi dev test. 
    """
    wiki_dir, tydi_dir = args.wiki_dir, args.tydi_dir
    output_dir = args.output_dir

    lang2doc2id = {}
    for lang in LANGS:
        print(f"*** processing {lang} ***")
        wiki_name = f"{lang_full2abbr[lang]}wiki.20190201.json" if lang != "thai" else f"{lang_full2abbr[lang]}wiki.20190101.json"
        wiki_fn = os_join(wiki_dir, wiki_name)
        title2_id_psgs = load_psg_dict_from_wiki_json(wiki_json=wiki_fn)
        prepare_dataset_from_tydi(
            lang, tydi_dir, wiki_psg_dict=title2_id_psgs, output_dir=output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
