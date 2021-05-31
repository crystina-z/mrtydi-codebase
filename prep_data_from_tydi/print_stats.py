"""
Print stats (No. uniq queries; No. duplicate queries; No. valid labels) from train and dev jsonl
"""
import os
from passage_level_dataset import jsonl_loader, LANGS, os_join

root_dir = "/scratch/czhang/xling/data/tydi/tydi_official/has_pos_label"

def print_stats_from_jsonl(jsonl_fn, lang):
    questions = set()
    n_valid_labels = 0
    for question, doc_title, passages, rel_indexes in jsonl_loader(jsonl_fn, expected_lang=lang): 
        questions.add(question)
        n_valid_labels += len(rel_indexes)

    print(f"\tNumber of Uniq Questions: {len(questions)}")
    print(f"\tNumber of Labels: {n_valid_labels}")
    return questions


for lang in LANGS:
    print(lang)
    # for set_name in ["dev"]:
    # for set_name in ["train"]:

    set_name = "train"
    print(f"*** {set_name} ***")
    jsonl_fn = os_join(root_dir, f"tydiqa-v1.0-{set_name}.{lang}.jsonl") 
    train_questions = print_stats_from_jsonl(jsonl_fn, lang=lang)

    set_name = "dev"
    print(f"*** {set_name} ***")
    jsonl_fn = os_join(root_dir, f"tydiqa-v1.0-{set_name}.{lang}.jsonl") 
    test_questions = print_stats_from_jsonl(jsonl_fn, lang=lang)

    overlap = train_questions & test_questions
    print("Question overlap between train and dev: ", len(overlap))