"""
Given prepared collection and benchmark, 
seperate (1) topic (2) qrels  
given folds
"""
import os
import sys
import json

from utils import LANGS

os_join = os.path.join


def keep_only(fn, qids_to_keep):
    """ Read from file `fn`, and return only the lines which starts with the qids in qids_to_keep """
    with open(fn) as f:
        for line in f:
            qid = line.strip().split()[0]
            if qid in qids_to_keep:
                yield line


def split_train_dev_set(root_dir):
    """ Split the topics and qrels file for each language into train and dev set, according to the folds.json """
    for lang in LANGS: 
        print(f"Processing {lang}...") 
        lang_dir = os_join(root_dir, lang)
        fold_fn = os_join(lang_dir, "folds.json")
        assert os.path.exists(fold_fn), f"Folds file unfound for langugage {lang}."

        folds = json.load(open(fold_fn))
        for n_dup in [len(set(folds["train"]) & set(folds["dev"])), len(set(folds["train"]) & set(folds["test"])), len(set(folds["dev"]) & set(folds["test"]))]:
            assert n_dup == 0, f"{n_dup} overlap queries detected for language {lang}."

        for set_name in folds:
            set_content = set(folds[set_name])

            topic_fn, topic_out_fn = \
                os_join(lang_dir, "topic.tsv"), os_join(lang_dir, f"topic.{set_name}.tsv")  
            qrel_fn, qrel_out_fn = \
                os_join(lang_dir, "qrels.txt"), os_join(lang_dir, f"qrels.{set_name}.txt")  

            assert os.path.exists(topic_fn), f"Topic file unfound for langugage {lang}."
            assert os.path.exists(qrel_fn), f"Qrel file unfound for langugage {lang}."

            with open(topic_out_fn, "w") as fout:
                for line in keep_only(topic_fn, qids_to_keep=set_content):
                    fout.write(line)

            with open(qrel_out_fn, "w") as fout:
                for line in keep_only(qrel_fn, qids_to_keep=set_content):
                    fout.write(line)


if __name__ == "__main__":
    root_dir = sys.argv[1]
    split_train_dev_set(root_dir)
