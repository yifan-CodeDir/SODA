import argparse
import os, sys
import random
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import torch
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from Macros import Macros
from preprocessing import utils
import pickle
import copy

# Primary method respnosible for tokenizing method and mutated line
def tokenize_str(method, test, tokenizer):
    tokens = [tokenizer.cls_token]
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
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

def subsample_mutants(mutants, tokenizer):
    new_mutants = []
    for ind, mutant in enumerate(mutants):
        mutant_src_lines = copy.deepcopy(mutant["src_lines"])
        embed, mask, index = tokenize_str(mutant["src_lines"], mutant["tst_lines"], tokenizer)
        if mutant["new_line"] == "":
            mutant_src_lines.pop(mutant["mut_src_line_no"])
        else:
            mutant_src_lines[mutant["mut_src_line_no"]] = mutant["new_line"]

        embed_mutant, mask_mutant, index_mutant = tokenize_str(mutant_src_lines, mutant["tst_lines"], tokenizer)

        new_mutant = {
            "embed": embed,
            "mask": mask,
            "index": index,
            "id": ind,
            "type": "orig",
            "label": 0
        }

        new_mutant_mutated = {
            "embed": embed_mutant,
            "mask": mask_mutant,
            "index": index_mutant,
            "id": ind,
            "type": "mutated",
            "label": mutant["label"]
        }
        new_mutants.append(new_mutant)
        new_mutants.append(new_mutant_mutated)

    return new_mutants

def map_src_to_dst(src_dir, dest_dir, tokenizer):
    for filepath in tqdm(os.listdir(src_dir)):
        with open(f"{src_dir}/{filepath}","rb") as sf:
            mutants = pickle.load(sf)
        
        with open(f"{dest_dir}/{filepath}", "wb") as df:
            mapped_mutants = subsample_mutants(mutants, tokenizer)
            pickle.dump(mapped_mutants, df)

def main(tokenizer):
    (Macros.defects4j_root_dir / "codebert" / "train").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / "codebert" / "val").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / "codebert" / "test").mkdir(exist_ok=True, parents=True)

    map_src_to_dst((Macros.defects4j_root_dir / "base_set" / "train"), (Macros.defects4j_root_dir / "codebert" / "train"), tokenizer)
    map_src_to_dst((Macros.defects4j_root_dir / "base_set" / "val"), (Macros.defects4j_root_dir / "codebert" / "val"), tokenizer)
    map_src_to_dst((Macros.defects4j_root_dir / "base_set" / "test"), (Macros.defects4j_root_dir / "codebert" / "test"), tokenizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codebert")

    args = parser.parse_args()

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])

    utils.set_seed(Macros.random_seed)
    main(tokenizer)
