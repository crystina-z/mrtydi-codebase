"""
This file converts the checkpoint generated from mDPR to huggingface format
"""

import os
import argparse

import torch
from torch.serialization import default_restore_location


def convert_context_model(inp_file, outp_dir):
    if outp_dir: 
        os.makedirs(outp_dir, exist_ok=True)

    print("Start converting context model...", end="")
    # state_dict = torch.load('dpr_biencoder.37', map_location=lambda s, l: default_restore_location(s, "cpu"))
    state_dict = torch.load(inp_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    encoder_state = {}
    for key, value in state_dict["model_dict"].items():
        if "ctx_model.encode_proj" in key:
            new_key = "ctx_encoder."+ key[len("ctx_model."):]
            encoder_state[new_key] = value
        elif key.startswith("ctx_model"):
            new_key = "ctx_encoder.bert_model."+ key[len("ctx_model."):]
            encoder_state[new_key] = value
    torch.save(
        encoder_state, 
        os.path.join(outp_dir, 'pytorch_model.bin'),
    )
    print("Finished")


def convert_question_model(inp_file, outp_dir):
    if outp_dir: 
        os.makedirs(outp_dir, exist_ok=True)

    print("Start converting question model...", end="")
    # state_dict = torch.load('dpr_biencoder.30.460', map_location=lambda s, l: default_restore_location(s, "cpu"))
    state_dict = torch.load(inp_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    encoder_state = {}
    for key, value in state_dict["model_dict"].items():
        if "question_model.encode_proj" in key:
            new_key = "question_encoder."+ key[len("question_model."):]
            encoder_state[new_key] = value
        elif key.startswith("question_model"):
            new_key = "question_encoder.bert_model."+ key[len("question_model."):]
            encoder_state[new_key] = value
    torch.save(
        encoder_state, 
        os.path.join(outp_dir, 'pytorch_model.bin'),
    )
    print("Finished")


def main(args):
    ctx_model_path = args.ctx_model_path
    q_model_path = args.q_model_path
    output_model_path = args.output_model_path

    if ctx_model_path:
        convert_context_model(inp_file=ctx_model_path, outp_dir=os.path.join(output_model_path, "ctx_model"))

    if q_model_path:
        convert_question_model(inp_file=ctx_model_path, outp_dir=os.path.join(output_model_path, "q_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Covnert ckpt weight from mDPR to hf")
    parser.add_argument("--ctx-model-path", "-ctx", default=None, type=str)
    parser.add_argument("--q-model-path", "-q", default=None, type=str)
    parser.add_argument("--output-model-path", "-o", default="./", type=str, help="directory to save the converted output models")

    args = parser.parse_args()
    main(args)
