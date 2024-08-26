import argparse
import difflib
import os, sys
import random
import numpy as np
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from Macros import Macros
from preprocessing import utils
import pickle
import copy
import time

# Tokenize with truncating long text
def tokenize_str_with_truncat(method, test, new_line, line_idx, tokenizer, max_length=512):
    tokens = [tokenizer.cls_token]
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        if i == line_idx:
            before_line = tokenizer.tokenize(method[i])
            after_line = tokenizer.tokenize(new_line)

            seqmatcher = difflib.SequenceMatcher(a=before_line, b=after_line, autojunk=False)
            for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
                if tag == "equal":
                    tokens += before_line[a0:a1]
                else:
                    tokens += ["<BEFORE>"]
                    tokens += before_line[a0:a1]
                    tokens += ["AFTER"]
                    tokens += after_line[b0:b1]
                    tokens += ["<ENDDIFF>"]
        else:
            tokens += tokenizer.tokenize(method[i])
    
    tokens += [tokenizer.sep_token]

    for i in range(len(test)):
        tokens += tokenizer.tokenize(test[i])
    tokens += [tokenizer.eos_token]

    if len(tokens) > 512:  # do truncation
        tokens = tokens[:max_length-1] + [tokenizer.eos_token]

    ids =  tokenizer.convert_tokens_to_ids(tokens) 
    mask = [1] * (len(tokens))
    
    padding_length = max_length - len(tokens)
    ids += [tokenizer.pad_token_id]*padding_length
    mask+=[0]*padding_length
    return ids, mask, 0

# Primary method respnosible for tokenizing method and mutated line
def tokenize_str(method, test, new_line, line_idx, tokenizer):
    tokens = [tokenizer.cls_token]
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        if i == line_idx:
            before_line = tokenizer.tokenize(method[i])
            after_line = tokenizer.tokenize(new_line)

            seqmatcher = difflib.SequenceMatcher(a=before_line, b=after_line, autojunk=False)
            for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
                if tag == "equal":
                    tokens += before_line[a0:a1]
                else:
                    tokens += ["<BEFORE>"]
                    tokens += before_line[a0:a1]
                    tokens += ["AFTER"]
                    tokens += after_line[b0:b1]
                    tokens += ["<ENDDIFF>"]
        else:
            tokens += tokenizer.tokenize(method[i])
    
    tokens += [tokenizer.sep_token]

    for i in range(len(test)):
        tokens += tokenizer.tokenize(test[i])
    tokens += [tokenizer.eos_token]

    ids =  tokenizer.convert_tokens_to_ids(tokens) 
    mask = [1] * (len(tokens))
    
    padding_length = 512 - len(tokens)
    ids += [tokenizer.pad_token_id]*padding_length
    mask+=[0]*padding_length
    return ids, mask, 0

def subsample_mutants(mutants, tokenizer, is_test):
    new_mutants = []
    for ind, suite in enumerate(mutants):
        suite_new = {"label": suite["label"], "mutants": []}
        for mutant in suite["mutants"]:
            if is_test:
                embed, mask, index = tokenize_str_with_truncat(mutant["src_lines"], mutant["tst_lines"], mutant["new_line"], mutant["mut_src_line_no"], tokenizer)
            else:
                embed, mask, index = tokenize_str(mutant["src_lines"], mutant["tst_lines"], mutant["new_line"], mutant["mut_src_line_no"], tokenizer)
            new_mutant = {
                "embed": embed,
                "mask": mask,
                "index": index,
                "label": mutant["label"]
            }
            suite_new["mutants"].append(new_mutant)
        new_mutants.append(suite_new)

    return new_mutants
    
def map_src_to_dst(src_dir, dest_dir, tokenizer, is_test=False):
    for filepath in tqdm(os.listdir(src_dir)):
        with open(f"{src_dir}/{filepath}","rb") as sf:
            mutants = pickle.load(sf)
        
        with open(f"{dest_dir}/{filepath}", "wb") as df:
            mapped_mutants = subsample_mutants(mutants, tokenizer, is_test)
            pickle.dump(mapped_mutants, df)

def main(args, tokenizer):    
    for proj_name, _ in Macros.train_versions.items():
        # (Macros.defects4j_root_dir / args.dst_set / proj_name / "train").mkdir(exist_ok=True, parents=True)
        (Macros.defects4j_root_dir / args.dst_set / proj_name / "val").mkdir(exist_ok=True, parents=True)
        (Macros.defects4j_root_dir / args.dst_set / proj_name / "test").mkdir(exist_ok=True, parents=True)

        # map_src_to_dst((Macros.defects4j_root_dir / args.src_set / proj_name / "train"), (Macros.defects4j_root_dir / args.dst_set / proj_name / "train"), tokenizer)
        map_src_to_dst((Macros.defects4j_root_dir / args.src_set / proj_name / "val"), (Macros.defects4j_root_dir / args.dst_set / proj_name / "val"), tokenizer)
        map_src_to_dst((Macros.defects4j_root_dir / args.src_set / proj_name / "test"), (Macros.defects4j_root_dir / args.dst_set / proj_name / "test"), tokenizer, is_test=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codebert")
    parser.add_argument("--src_set", help="source set", default="cross_version_suite_base_set")
    parser.add_argument("--dst_set", help="dest set", default="cross_version_suite_codebert_token_diff")

    args = parser.parse_args()

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    # tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])
    tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--microsoft--codebert-base", local_files_only=True)  # locally load codebert

    utils.set_seed(Macros.random_seed)
    main(args, tokenizer)
