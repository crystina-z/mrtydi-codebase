"""
Step 1 in the open-retrieval data preparation.
Given primary tasks jsonl downloaded from `https://github.com/google-research-datasets/tydiqa`, 
this file separate the `lines` "with_answer" from the ones "without_answer",
where the "without_answer" lines contain no valid answer in the `annotation` field of the jsonl file.
Note that "lines containing the answer" is different from "questions having answer", 
since there can be duplciate quesitons cross the lines. 
That is, the processed file under `$tydi_dir/with_answer` may still contain the duplicate questions. 
"""
import os
import sys
import json


def have_answer(line):
    line = json.loads(line)
    doc = line["document_plaintext"]
    candidates = line["passage_answer_candidates"]
    for annotation in line["annotations"]:
        can_id = annotation["passage_answer"]["candidate_index"]
        if can_id != -1:
            span = candidates[can_id]
            passage = doc[span["plaintext_start_byte"]:span["plaintext_end_byte"] + 1]
            if passage == "":
                continue
            return True
    return False



def get_anserable_questions(tydi_dir):
    tydi_with_answer_dir=f"{tydi_dir}/with_answer" 
    tydi_without_answer_dir=f"{tydi_dir}/without_answer"

    os.makedirs(tydi_with_answer_dir, exist_ok=True)
    os.makedirs(tydi_without_answer_dir, exist_ok=True)

    for set_name in ["train", "dev"]:
        n_total, n_rel = 0, 0
        empty_answer = 0
        inp_fn = f"{tydi_dir}/tydiqa-v1.0-{set_name}.jsonl"
        with_answer_fn = f"{tydi_with_answer_dir}/tydiqa-v1.0-{set_name}.jsonl" 
        without_answer_fn = f"{tydi_without_answer_dir}/tydiqa-v1.0-{set_name}.jsonl" 

        with open(inp_fn) as fin, open(with_answer_fn, "w") as f_ans, open(without_answer_fn, "w") as f_unans:
            for line in fin:
                if have_answer(line):
                    f_ans.write(line) 
                    n_rel += 1
                else:
                    f_unans.write(line)
                n_total += 1
        print(f"{n_rel} out of {n_total} lines have answers in {set_name}")
        # print(f"{empty_answer} answers are empty.")
    

if __name__ == "__main__":
    tydi_dir = sys.argv[1]
    get_anserable_questions(tydi_dir=tydi_dir)
