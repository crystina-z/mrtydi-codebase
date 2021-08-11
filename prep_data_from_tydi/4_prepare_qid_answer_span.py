"""
Given the original Tydi jsonl and the prepared topic.tsv,
generate a json file with each line to be:
{"qid": qid, "question": query, "answers": [answer1, answer2, ...]}
"""
import os
import sys
import json
from argparse import ArgumentParser

os_join = os.path.join
os_dir = os.path.dirname
PACKAGE_PATH = os_dir(os_dir(os.path.abspath(__file__)))
sys.path.append(PACKAGE_PATH)
print(PACKAGE_PATH)

from utils import lang_full2abbr, LANGS, load_topic_tsv, byte_str, byte_slice


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--tydi_dir", type=str, required=True, 
        help="The directory containing the language-grouped TyDi train and dev jsonl files.")
    parser.add_argument(
        "--mrtydi_dir", type=str, required=True, 
        help="The directory containing the prepared Mr.TyDi data, and also the directory where the output .json file would be stored")

    return parser.parse_args()


def jsonl_loader(jsonl_path, expected_lang):
    with open(jsonl_path) as f:
        for line in f:
            line = json.loads(line)
            question = line["question_text"]
            assert expected_lang == line["language"], f"expect {expected_lang} but got {line['language']}"

            doc = line["document_plaintext"]
            answer_spans = [(
                annotation["minimal_answer"]["plaintext_start_byte"], annotation["minimal_answer"]["plaintext_end_byte"]) 
                for annotation in line["annotations"]]
            answers = {
                # " ".join(doc[start:end + 1].replace("\n", " ").split())
                byte_slice(doc, start, end) 
                for start, end in answer_spans if not (start == -1 and end == -1)}

            yield question, list(sorted(answers))


def main(args):
    tydi_dir = args.tydi_dir
    mrtydi_dir = args.mrtydi_dir

    for lang in LANGS:
        tydi_jsonl_path = os_join(tydi_dir, f"tydiqa-v1.0-dev.{lang}.jsonl")
        topic_fn = os_join(mrtydi_dir, lang, "topic.tsv")
        qid2answers_jsonl_path = os_join(mrtydi_dir, lang, f"qid2answers.{lang}.jsonl")

        id2topic = load_topic_tsv(topic_fn)
        topic2id = {topic: id for id, topic in id2topic.items()}

        with open(qid2answers_jsonl_path, "w") as f:
            for question, answers in jsonl_loader(tydi_jsonl_path, expected_lang=lang):
                if question not in topic2id:
                    raise ValueError(f"{question} was not included in the {topic_fn}.")

                qid = topic2id[question]
                line = json.dumps({"qid": qid, "question": question, "answers": answers})
                f.write(line + "\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
