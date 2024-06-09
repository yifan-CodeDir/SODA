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

# Method that splits list into train validation and test sets
def split_two(lst, test_r):
    random.shuffle(lst)

    indicies_for_splitting = [int(len(lst) * test_r)]
    test, val = np.split(lst, indicies_for_splitting)
    return test, val

def split_cp_fold(mutants, test_map, set_name, fold_num, codet5_tokenizer, codebert_tokenizer):
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

    tmp_mut = []
    tmp_mut_pos = []
    index = 0
    for p in train_proj: # process all subject
        mut, mut_pos = subsample_mutants_with_pos(mutants[p], test_map, codet5_tokenizer, codebert_tokenizer)
        # while len(mut_pos) < len(mut):  # oversampling
        #     mut_pos.append(random.choice(mut_pos))

        mut = tmp_mut + mut
        mut_pos = tmp_mut_pos + mut_pos

        for i in range(0, len(mut) - len(mut) % 10_000, 10_000):
            with open(Macros.defects4j_root_dir / set_name / "train" / f"train_{str(index)}", "wb") as f:
                pickle.dump(mut[i:i+10_000], f) 
            with open(Macros.defects4j_root_dir / set_name / "train_pos" / f"train_{str(index)}", "wb") as f:
                pickle.dump(mut_pos[i:i+10_000], f) 
            index += 1

        if (len(mut) % 10_000) != 0:
            tmp_mut = mut[-(len(mut) % 10_000):]   # udpate tmp
            tmp_mut_pos = mut_pos[-(len(mut) % 10_000):]
        else:
            tmp_mut = []
            tmp_mut_pos = []
            
    if len(tmp_mut) > 0 :
        with open(Macros.defects4j_root_dir / set_name / "train" / f"train_{index}", "wb") as f:
            pickle.dump(tmp_mut, f) 
        with open(Macros.defects4j_root_dir / set_name / "train_pos" / f"train_{index}", "wb") as f:
            pickle.dump(tmp_mut_pos, f)

    cp_test = []
    for p in test_proj:
        cp_test += mutants[p]
    
    cp_val = []
    for p in val_proj:
        cp_val += mutants[p]
    
    subsample_test_mutants(cp_test, test_map, codet5_tokenizer, codebert_tokenizer, "test", set_name)
    subsample_mutants(cp_val, test_map, codet5_tokenizer, codebert_tokenizer, "val", set_name)


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

def subsample_mutants_with_pos(mutants, test_map, codet5_tokenizer, codebert_tokenizer):
    new_mutants = []
    new_pos_mutants_dict_mut_no_key = {}
    new_pos_mutants_dict_method_key = {}
    new_pos_mutants = []
    i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

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
                "class_and_method": mutant["class_name"] + ":" + mutant["method_name"],
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
            new_mutants.append(new_mutant)
            if new_mutant["label"] == 1:   # collect positive sample for all mutants
                if new_mutant["mut_no"] in new_pos_mutants_dict_mut_no_key.keys():  # use mutant_no as key
                    new_pos_mutants_dict_mut_no_key[new_mutant["mut_no"]].append(new_mutant)
                else:
                    new_pos_mutants_dict_mut_no_key[new_mutant["mut_no"]] = [new_mutant]

                if new_mutant["class_and_method"] in new_pos_mutants_dict_method_key.keys():     # use method_name as key
                    new_pos_mutants_dict_method_key[new_mutant["class_and_method"]].append(new_mutant)
                else:
                    new_pos_mutants_dict_method_key[new_mutant["class_and_method"]] = [new_mutant]

    # sample positive samples for the same mutants
    for mutant in new_mutants:
        if random.random() < 0.2:  # sample difficult sample with 20% probability 
            if mutant["mut_no"] in new_pos_mutants_dict_mut_no_key.keys(): # if this mutant has positive sample
                new_pos_mutants.append(random.choice(new_pos_mutants_dict_mut_no_key[mutant["mut_no"]]))
            elif mutant["class_and_method"] in new_pos_mutants_dict_method_key.keys():  # if mutants of same method has positive sample
                new_pos_mutants.append(random.choice(new_pos_mutants_dict_method_key[mutant["class_and_method"]]))
            else:  # else randomly choose a mutant-test pair
                new_pos_mutants.append(random.choice(random.choice(list(new_pos_mutants_dict_mut_no_key.values()))))
        else:
            new_pos_mutants.append(random.choice(random.choice(list(new_pos_mutants_dict_mut_no_key.values()))))

    return new_mutants, new_pos_mutants
    #         if len(new_mutants) == 10_000:
    #             # random.shuffle(new_mutants)   # move the random shuffle to split part
    #             with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
    #                 pickle.dump(new_mutants, f) 
                
    #             new_mutants = []
    #             i += 1

    # # random.shuffle(new_mutants)   # move the random shuffle to split part
    # print(num_skipped)
    # print(i * 10_000 + len(new_mutants))
    # with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
    #     pickle.dump(new_mutants, f) 

def subsample_mutants(mutants, test_map, codet5_tokenizer, codebert_tokenizer, prefix, set_name):
    new_mutants = []
    i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

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
            new_mutants.append(new_mutant)

            if len(new_mutants) == 10_000:
                random.shuffle(new_mutants)   # move the random shuffle to split part
                with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
                    pickle.dump(new_mutants, f) 
                new_mutants = []
                i += 1

    random.shuffle(new_mutants)   # move the random shuffle to split part
    print(num_skipped)
    print(i * 10_000 + len(new_mutants))
    with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix}_{str(i)}", "wb") as f:
        pickle.dump(new_mutants, f) 


def subsample_test_mutants(mutants, test_map, codet5_tokenizer, codebert_tokenizer, prefix, set_name):  # do not skip sample whose length > 512
    new_mutants = []
    i = 0
    num_skipped = 0
    key_set = set()
    for ind, mutant in enumerate(tqdm(mutants)):
        key = str(mutant["mut_src_line_no"])+mutant["before"]+mutant["after"]+mutant["class_name"]+mutant["method_name"]
        if key in key_set:
            continue
        key_set.add(key)

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
            new_mutants.append(new_mutant)

            if len(new_mutants) == 10_000:
                random.shuffle(new_mutants)   # move the random shuffle to split part
                with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix.split('_')[0]}_{str(i)}", "wb") as f:
                    pickle.dump(new_mutants, f) 
                new_mutants = []
                i += 1

    random.shuffle(new_mutants)   # move the random shuffle to split part
    print(num_skipped)
    print(i * 10_000 + len(new_mutants))
    with open(Macros.defects4j_root_dir / set_name / f"{prefix}" / f"{prefix.split('_')[0]}_{str(i)}", "wb") as f:
        pickle.dump(new_mutants, f)

def main(args, codet5_tokenizer, codebert_tokenizer):
    with open(args.mutants_file, "rb") as f:
        mutants = pickle.load(f) 

    with open(args.test_file, "rb") as f:
        test_map = pickle.load(f) 
    
    fold_num = 5

    set_name = f"cross_project_soda_fold{fold_num}"

    (Macros.defects4j_root_dir / set_name / "train").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / set_name / "train_pos").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / set_name / "val").mkdir(exist_ok=True, parents=True)
    # (Macros.defects4j_root_dir / set_name / "test").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / set_name / "test").mkdir(exist_ok=True, parents=True)  ##### 

    split_cp_fold(mutants, test_map, set_name, fold_num, codet5_tokenizer, codebert_tokenizer) 

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants_file", help="where mutants are located", default=Macros.defects4j_root_dir / "mutants_new_cp.pkl")
    parser.add_argument("--test_file", help="where test mapping is located", default=Macros.defects4j_root_dir / "test_map_new_cp.pkl")

    parser.add_argument("--train_percentage", help="train data split", type=int, default=Macros.default_train_percentage)
    parser.add_argument("--validation_percentage", help="validation data split", type=int, default=Macros.default_validation_percentage)
    parser.add_argument("--test_percentage", help="test data split", type=int, default=Macros.default_test_percentage)
    parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codebert")
    # parser.add_argument("--is_cp", help="whether cross project or not", action="store_true")

    args = parser.parse_args()
    # if args.is_cp:
    #     print("NOTE: you must set mutants_file and test_file manually when using this option")

    tokenizer_dict = Macros.MODEL_DICT[args.model]
    # tokenizer = tokenizer_dict["tokenizer"].from_pretrained(tokenizer_dict["pretrained"])
    codet5_tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--salesforce--codet5-base", local_files_only=True)  # locally load codet5
    codebert_tokenizer = tokenizer_dict["tokenizer"].from_pretrained("/xxx/huggingface/models--microsoft--codebert-base", local_files_only=True)  # locally load code-bert

    utils.set_seed(Macros.random_seed)
    main(args, codet5_tokenizer, codebert_tokenizer)
