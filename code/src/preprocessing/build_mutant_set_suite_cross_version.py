import argparse
import os, sys
import random
import numpy as np
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from Macros import Macros
from preprocessing import utils
import pickle

# Method that splits list into train validation and test sets
def split_three(lst, args):
    train_r, val_r, test_r = args.train_percentage/100, args.validation_percentage/100, args.test_percentage/100

    indicies_for_splitting = [int(len(lst) * train_r), int(len(lst) * (train_r+val_r))]
    train, val, test = np.split(lst, indicies_for_splitting)
    return train, val, test

# Method that randomly samples and splits list of mutants
def random_sample_mutants(mutants, args):
    random.shuffle(mutants)
    train, val, test = split_three(mutants, args)

    return train, val, test

def split_cp_fold(mutants, fold_num):
    # projects = ["Chart", "Csv", "Gson", "JacksonCore", "Lang"]
    if fold_num == 1:
        train_proj = ["Gson", "JacksonCore", "Lang"]
        test_proj = ["Chart"]
        val_proj = ["Csv"]
    elif fold_num == 2:
        train_proj = ["Chart", "JacksonCore", "Gson"]
        test_proj = ["Csv"]
        val_proj = ["Lang"]
    elif fold_num == 3:
        train_proj = ["Chart", "JacksonCore", "Lang"]
        test_proj = ["Gson"]
        val_proj = ["Csv"]
    elif fold_num == 4:
        train_proj = ["Chart", "Lang", "Gson"]
        test_proj = ["JacksonCore"]
        val_proj = ["Csv"]
    elif fold_num == 5:
        train_proj = ["Chart", "JacksonCore", "Gson"]
        test_proj = ["Lang"]
        val_proj = ["Csv"]
    
    cp_train = []
    for p in train_proj:
        cp_train += mutants[p]

    cp_test = []
    for p in test_proj:
        cp_test += mutants[p]
    
    cp_val = []
    for p in val_proj:
        cp_val += mutants[p]

    return cp_train, cp_val, cp_test

# Primary method respnosible for tokenizing method and mutated line
def tokenize_str(method, test, new_line, line_idx, tokenizer):
    tokens = [tokenizer.cls_token]
    # Add method tokens to list with seperator token between lines
    for i in range(len(method)):
        if i == line_idx:
            tokens += ["<BEFORE>"]
            tokens += tokenizer.tokenize(method[i])
            tokens += ["<AFTER>"]
            tokens += tokenizer.tokenize(new_line)
            tokens += ["<ENDDIFF>"]
        else:
            tokens += tokenizer.tokenize(method[i])
    
    tokens += [tokenizer.sep_token]

    for i in range(len(test)):
        tokens += tokenizer.tokenize(test[i])
    tokens += [tokenizer.eos_token]

    ids =  tokenizer.convert_tokens_to_ids(tokens) 
    mask = [1] * (len(tokens))
    
    if len(tokens) > 512:
        return [], [], -1

    padding_length = 512 - len(tokens)
    ids += [tokenizer.pad_token_id]*padding_length
    mask+=[0]*padding_length
    return ids, mask, 0

def subsample_mutants(mutants, test_map, new_mutants, i, codet5_tokenizer, codebert_tokenizer, proj, set_name, prefix):
    # new_mutants = []
    # i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

        curr_suite = {"label": mutant["label"], "mutants": []}
        for test in mutant["tests"]:
            if mutant["mut_src_line_no"] >= len(mutant["src_lines"]):
                continue
            
            mutated_src_lines = mutant["src_lines"]
            new_line = None
            
            if "NOOP" in mutant["after"]:
                if mutant["mutator"] != "STD":
                    print(mutant)
                else:
                    new_line = ""
            else:        
                new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(mutant["before"], mutant["after"], 1)
                no_space_before = mutant["before"].replace(" ", "")
                lowercase_before = mutant["before"].replace("F", "f")
                no_space_after = mutant["after"].replace(" ", "")
                lowercase_after = mutant["after"].replace("F", "f")

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(no_space_before, no_space_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(lowercase_before, lowercase_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = None

            if new_line is None:
                continue
            
            codet5_embed, _, codet5_index = tokenize_str(mutated_src_lines, test_map[test], new_line, mutant["mut_src_line_no"], codet5_tokenizer)
            if codet5_index == -1:  # skip the token sequence whose length > 512 
                continue
            cls_token_id = 1
            if codet5_embed[codet5_index] != cls_token_id:
                num_skipped += 1
                continue

            codebert_embed, _, codebert_index = tokenize_str(mutated_src_lines, test_map[test], new_line, mutant["mut_src_line_no"], codebert_tokenizer)
            if codebert_index == -1:  # skip the token sequence whose length > 512 
                continue
            cls_token_id = 0
            if codebert_embed[codebert_index] != cls_token_id:
                num_skipped += 1
                continue

            new_mutant = {
                "mut_no": mutant["mut_no"],
                "test_method": test,
                "source_method": mutant["method_name"],
                "src_lines": mutant["src_lines"],
                "new_line": new_line,
                "tst_lines": test_map[test],
                "before_pmt": mutant["before_pmt"],
                "after_pmt": mutant["after_pmt"],
                "mutator": mutant["mutator"],
                "mut_src_line_no": mutant["mut_src_line_no"],
                "label": 1 if test in mutant["killing_tests"] else 0
            }
            curr_suite["mutants"].append(new_mutant)
        
        if len(curr_suite["mutants"]) > 0:
            random.shuffle(curr_suite["mutants"])
            new_mutants.append(curr_suite)

        if len(new_mutants) == 200:
            random.shuffle(new_mutants)
            # if model == "codet5":
            #     set_name = "codet5_base_set_suite_cp" if args.is_cp else "codet5_base_set_suite"
            # else:
            #     set_name = "base_set_suite_cp" if args.is_cp else "base_set_suite"
            with open(Macros.defects4j_root_dir / set_name / proj / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
                pickle.dump(new_mutants, f) 
            new_mutants = []
            i += 1

    return new_mutants, i
    # random.shuffle(new_mutants)
    # print(num_skipped)
    # print(i * 10_000 + len(new_mutants))
    # if model == "codet5":
    #     set_name = "codet5_base_set_suite_cp" if args.is_cp else "codet5_base_set_suite"
    # else:
    #     set_name = "base_set_suite_cp" if args.is_cp else "base_set_suite"
    # with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
    #     pickle.dump(new_mutants, f) 

def subsample_test_mutants(mutants, test_map, new_mutants, i, codet5_tokenizer, codebert_tokenizer, proj, set_name, prefix):
    # new_mutants = []
    # i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

        curr_suite = {"label": mutant["label"], "mutants": []}
        for test in mutant["tests"]:
            if mutant["mut_src_line_no"] >= len(mutant["src_lines"]):
                continue
            
            mutated_src_lines = mutant["src_lines"]
            new_line = None
            
            if "NOOP" in mutant["after"]:
                if mutant["mutator"] != "STD":
                    print(mutant)
                else:
                    new_line = ""
            else:        
                new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(mutant["before"], mutant["after"], 1)
                no_space_before = mutant["before"].replace(" ", "")
                lowercase_before = mutant["before"].replace("F", "f")
                no_space_after = mutant["after"].replace(" ", "")
                lowercase_after = mutant["after"].replace("F", "f")

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(no_space_before, no_space_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = mutated_src_lines[mutant["mut_src_line_no"]].replace(lowercase_before, lowercase_after, 1)

                if new_line == mutant["src_lines"][mutant["mut_src_line_no"]]:
                    new_line = None

            if new_line is None:
                continue
            
            # codet5_embed, _, codet5_index = tokenize_str(mutated_src_lines, test_map[test], new_line, mutant["mut_src_line_no"], codet5_tokenizer)
            # if codet5_index == -1:  # skip the token sequence whose length > 512 
            #     continue
            # cls_token_id = 1
            # if codet5_embed[codet5_index] != cls_token_id:
            #     num_skipped += 1
            #     continue

            # codebert_embed, _, codebert_index = tokenize_str(mutated_src_lines, test_map[test], new_line, mutant["mut_src_line_no"], codebert_tokenizer)
            # if codebert_index == -1:  # skip the token sequence whose length > 512 
            #     continue
            # cls_token_id = 0
            # if codebert_embed[codebert_index] != cls_token_id:
            #     num_skipped += 1
            #     continue

            new_mutant = {
                "mut_no": mutant["mut_no"],
                "test_method": test,
                "source_method": mutant["method_name"],
                "src_lines": mutant["src_lines"],
                "new_line": new_line,
                "tst_lines": test_map[test],
                "before_pmt": mutant["before_pmt"],
                "after_pmt": mutant["after_pmt"],
                "mutator": mutant["mutator"],
                "mut_src_line_no": mutant["mut_src_line_no"],
                "label": 1 if test in mutant["killing_tests"] else 0
            }
            curr_suite["mutants"].append(new_mutant)
        
        if len(curr_suite["mutants"]) > 0:
            random.shuffle(curr_suite["mutants"])
            new_mutants.append(curr_suite)

        if len(new_mutants) == 200:
            random.shuffle(new_mutants)
            # if model == "codet5":
            #     set_name = "codet5_base_set_suite_cp" if args.is_cp else "codet5_base_set_suite"
            # else:
            #     set_name = "base_set_suite_cp" if args.is_cp else "base_set_suite"
            with open(Macros.defects4j_root_dir / set_name / proj / f"{prefix}" / f"{prefix.split('_')[0]}_{str(i)}", "wb") as f:
                pickle.dump(new_mutants, f) 
            new_mutants = []
            i += 1

    return new_mutants, i
    # random.shuffle(new_mutants)
    # print(num_skipped)
    # print(i * 10_000 + len(new_mutants))
    # if model == "codet5":
    #     set_name = "codet5_base_set_suite_cp" if args.is_cp else "codet5_base_set_suite"
    # else:
    #     set_name = "base_set_suite_cp" if args.is_cp else "base_set_suite"
    # with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix.split('_')[0]}_{str(i)}", "wb") as f:
    #     pickle.dump(new_mutants, f) 

def main(args, codet5_tokenizer, codebert_tokenizer, subset):

    set_name = f"cross_version_suite_base_set"

    if subset == "train":
        version_dict = Macros.train_versions
    elif subset == "test":
        version_dict = Macros.test_versions
    elif subset == "val":
        version_dict = Macros.valid_versions
    else:
        raise Exception("Unknown subset!")

    for proj_name, proj_no_list in version_dict.items():
        print(f"Processing {subset} set of {proj_name} ...")
        (Macros.defects4j_root_dir / set_name / proj_name / subset).mkdir(exist_ok=True, parents=True)
        new_mutants = []  # initialize variables for recording
        i = 0
        for proj_no in proj_no_list:
            with open(args.mutants_dir / proj_name / str(proj_no) / "mutants.pkl", "rb") as f:
                mutants = pickle.load(f) 

            with open(args.test_dir / proj_name / str(proj_no) / "test_map.pkl", "rb") as f:
                test_map = pickle.load(f) 

            random.shuffle(mutants)
            if subset == "test":
                new_mutants, i = subsample_test_mutants(mutants, test_map, new_mutants, i, codet5_tokenizer, codebert_tokenizer, proj_name, set_name, subset)
            else:
                new_mutants, i = subsample_mutants(mutants, test_map, new_mutants, i, codet5_tokenizer, codebert_tokenizer, proj_name, set_name, subset)

            print(f"# of mutants for {proj_name}-{proj_no}: {i * 200 + len(new_mutants)}")
        
        print(f"#### In total: {subset} set of {proj_name} has {i * 200 + len(new_mutants)} samples ####")
        # output the rest new_mutants
        random.shuffle(new_mutants)
        with open(Macros.defects4j_root_dir / set_name / proj_name / f"{subset}" / f"{subset.split('_')[0]}_{str(i)}", "wb") as f:
            pickle.dump(new_mutants, f) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants_dir", help="where mutants are located", default=Macros.defects4j_root_dir / "cross_version_raw")
    parser.add_argument("--test_dir", help="where test mapping is located", default=Macros.defects4j_root_dir / "cross_version_raw")

    parser.add_argument("--train_percentage", help="train data split", type=int, default=Macros.default_train_percentage)
    parser.add_argument("--validation_percentage", help="validation data split", type=int, default=Macros.default_validation_percentage)
    parser.add_argument("--test_percentage", help="test data split", type=int, default=Macros.default_test_percentage)
    parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codebert")
    # parser.add_argument("--is_cp", help="whether cross project or not", action="store_true")

    args = parser.parse_args()

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    # tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])
    codet5_tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--salesforce--codet5-base", local_files_only=True)  # locally load codet5
    codebert_tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--microsoft--codebert-base", local_files_only=True)  # locally load code-bert

    utils.set_seed(Macros.random_seed)
    main(args, codet5_tokenizer, codebert_tokenizer, "train")
    main(args, codet5_tokenizer, codebert_tokenizer, "val")
    main(args, codet5_tokenizer, codebert_tokenizer, "test")
