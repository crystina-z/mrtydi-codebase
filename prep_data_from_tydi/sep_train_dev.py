"""
Given prepared collection and benchmark, 
seperate (1) topic (2) qrels  
given folds
"""
import os
import sys
import json

os_join = os.path.join

# root_dir = sys.argv[1]
root_dir = "/scratch/czhang/xling/data/tydi/open-retrieval" 
# root_dir = "/scratch/czhang/xling/data/tydi/open-retrieval/test" 

def keep_only(fn, qids_to_keep):
    with open(fn) as f:
        for line in f:
            qid = int(line.strip().split()[0])
            if qid in qids_to_keep:
                yield line


# LANGS = "arabic  bengali  finnish  indonesian  japanese  korean  swahili  telugu  thai russian".split() 
LANGS = "english".split() 
# LANGS = "thai".split() 
for lang in LANGS: 
    print(f"*** Processing {lang} ***")
    lang_dir = os_join(root_dir, lang)
    fold_fn = os_join(lang_dir, "folds.json")
    if not os.path.exists(fold_fn):
        continue

    folds = json.load(open(fold_fn))

    for set_name in folds:
        set_content = set(folds[set_name])

        topic_fn, topic_out_fn = \
            os_join(lang_dir, "topic.tsv"), os_join(lang_dir, f"topic.{set_name}.tsv")  
        qrel_fn, qrel_out_fn = \
            os_join(lang_dir, "qrels.txt"), os_join(lang_dir, f"qrels.{set_name}.txt")  

        with open(topic_out_fn, "w") as fout:
            for line in keep_only(topic_fn, set_content):
                fout.write(line)

        with open(qrel_out_fn, "w") as fout:
            for line in keep_only(qrel_fn, set_content):
                fout.write(line)
