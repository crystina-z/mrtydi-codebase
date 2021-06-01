import os
import sys
import json

tydi_dir=sys.argv[1]
tydi_has_pos_label_dir=f"{tydi_dir}/has_pos_label"
tydi_has_no_pos_label_dir=f"{tydi_dir}/no_pos_label"

os.makedirs(tydi_has_pos_label_dir, exist_ok=True)
os.makedirs(tydi_has_no_pos_label_dir, exist_ok=True)


# global empty_answer
# empty_answer = 0

def has_pos_label(line):
    global empty_answer 
    line = json.loads(line)
    doc = line["document_plaintext"]
    candidates = line["passage_answer_candidates"]
    for annotation in line["annotations"]:
        can_id = annotation["passage_answer"]["candidate_index"]
        if can_id != -1:
            span = candidates[can_id]
            # passage = " ".join(doc[span["plaintext_start_byte"]:span["plaintext_end_byte"] + 1].replace("\n", " ").split()).strip()
            passage = doc[span["plaintext_start_byte"]:span["plaintext_end_byte"] + 1]
            if passage == "": # or len(passage.split()) < 5:
                empty_answer += 1 
                continue
                # import pdb
                # pdb.set_trace()
            return True
    return False


for set_name in ["train", "dev"]:
    n_total, n_rel = 0, 0
    empty_answer = 0
    inp_fn = f"{tydi_dir}/tydiqa-v1.0-{set_name}.jsonl"
    outp_fn = f"{tydi_has_pos_label_dir}/tydiqa-v1.0-{set_name}.jsonl" 
    outp_fn_2 = f"{tydi_has_no_pos_label_dir}/tydiqa-v1.0-{set_name}.jsonl" 

    # with open(inp_fn) as fin, open(outp_fn, "w") as fout, open(outp_fn_2, "w") as fout2:
    with open(inp_fn) as fin: 
        for line in fin:
            if has_pos_label(line):
                # fout.write(line) 
                n_rel += 1
            else:
                # fout2.write(line)
                pass
            n_total += 1
    print(f"{n_rel} out of {n_total} questions have answers in {set_name}")
    print(f"{empty_answer} answers are empty.")
    