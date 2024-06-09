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

def get_context(source_line_tokens, line_idx, source_len):
    new_tokens = source_line_tokens[line_idx]  # the mutated line
    curr_window = 1
    # iteratively enlarge the window size until the length is larger then max_length
    while len(new_tokens) < source_len and ((line_idx - curr_window >= 1) or (line_idx + curr_window) < len(source_line_tokens)):
        if line_idx - curr_window >= 1:
            new_tokens = source_line_tokens[line_idx-curr_window] + new_tokens 
        if len(new_tokens) < source_len and line_idx + curr_window < len(source_line_tokens):
            new_tokens = new_tokens + source_line_tokens[line_idx+curr_window]
        curr_window += 1
    # index = new_index
    if len(new_tokens) > source_len: # truncate
        new_tokens = new_tokens[:source_len]
    return new_tokens

# Tokenize with a window surrounding the mutated line 
def tokenize_str_with_window(method, test, new_line, line_idx, tokenizer, max_length=512):
    tokens = [tokenizer.cls_token]
    source_tokens = []
    source_line_tokens = []  # record each line of tokens, list of list
    test_tokens = []
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        if i == line_idx:
            before_line = tokenizer.tokenize(method[i])
            after_line = tokenizer.tokenize(new_line)
            tmp_line_tokens = []

            seqmatcher = difflib.SequenceMatcher(a=before_line, b=after_line, autojunk=False)
            for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
                if tag == "equal":
                    source_tokens += before_line[a0:a1]
                    tmp_line_tokens += before_line[a0:a1]
                else:
                    source_tokens += ["<BEFORE>"]
                    source_tokens += before_line[a0:a1]
                    source_tokens += ["AFTER"]
                    source_tokens += after_line[b0:b1]
                    source_tokens += ["<ENDDIFF>"]

                    tmp_line_tokens += ["<BEFORE>"]
                    tmp_line_tokens += before_line[a0:a1]
                    tmp_line_tokens += ["AFTER"]
                    tmp_line_tokens += after_line[b0:b1]
                    tmp_line_tokens += ["<ENDDIFF>"]

            source_line_tokens.append(tmp_line_tokens)
        else:
            source_tokens += tokenizer.tokenize(method[i])
            source_line_tokens.append(tokenizer.tokenize(method[i]))

    for i in range(len(test)):
        test_tokens += tokenizer.tokenize(test[i])

    tokens = tokens + source_tokens + [tokenizer.sep_token] + test_tokens + [tokenizer.eos_token]

    if len(tokens) > max_length:  # truncate tokens
        tokens = [tokenizer.cls_token]
        source_len = int((max_length-3)/2)  # minus 3, considering cls, sep, and eos token
        if len(test_tokens) > int((max_length-3)/2): # truncate test tokens if length > (max_length-3)/2
            test_tokens = test_tokens[:int((max_length-3)/2)]
        else:
            source_len = max_length - 3 - len(test_tokens)
        # get a contextual window for source tokens: method signature + a context window around the mutated line
        source_tokens = source_line_tokens[0]
        source_len -= len(source_line_tokens[0])
        context_tokens = get_context(source_line_tokens, line_idx, source_len)
        source_tokens += context_tokens
        tokens = tokens + source_tokens + [tokenizer.sep_token] + test_tokens + [tokenizer.eos_token]
    
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
                embed, mask, index = tokenize_str_with_window(mutant["src_lines"], mutant["tst_lines"], mutant["new_line"], mutant["mut_src_line_no"], tokenizer)
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
    # (Macros.defects4j_root_dir / args.dst_set / "train").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / args.dst_set / "val").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / args.dst_set / "test").mkdir(exist_ok=True, parents=True)

    # map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "train"), (Macros.defects4j_root_dir / args.dst_set / "train"), tokenizer)
    map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "val"), (Macros.defects4j_root_dir / args.dst_set / "val"), tokenizer)
    map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "test"), (Macros.defects4j_root_dir / args.dst_set / "test"), tokenizer, is_test=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codet5")
    parser.add_argument("--src_set", help="source set", default="codet5_base_set_suite")
    parser.add_argument("--dst_set", help="dest set", default="codet5_token_diff_suite")

    args = parser.parse_args()

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    # tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])
    tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--salesforce--codet5-base", local_files_only=True)  # locally load codet5

    utils.set_seed(Macros.random_seed)
    main(args, tokenizer)
