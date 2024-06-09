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

def split_cp_fold(mutants, test_map, set_name, fold_num):
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
    
    subsample_mutants(cp_train, test_map, "train", set_name)
    subsample_mutants(cp_test, test_map, "test", set_name)
    subsample_mutants(cp_val, test_map, "val", set_name)


def subsample_mutants(mutants, test_map, prefix, set_name):
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

def main(args):
    with open(args.mutants_file, "rb") as f:
        mutants = pickle.load(f) 

    with open(args.test_file, "rb") as f:
        test_map = pickle.load(f) 
    
    for fold_num in range(1,6):

        set_name = f"seshat_cross_project_base_set_fold{fold_num}"

        (Macros.defects4j_root_dir / set_name / "train").mkdir(exist_ok=True, parents=True)
        # (Macros.defects4j_root_dir / set_name / "train_pos").mkdir(exist_ok=True, parents=True)
        (Macros.defects4j_root_dir / set_name / "val").mkdir(exist_ok=True, parents=True)
        (Macros.defects4j_root_dir / set_name / "test").mkdir(exist_ok=True, parents=True)

        split_cp_fold(mutants, test_map, set_name, fold_num) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutants_file", help="where mutants are located", default=Macros.defects4j_root_dir / "mutants_new_cp.pkl")
    parser.add_argument("--test_file", help="where test mapping is located", default=Macros.defects4j_root_dir / "test_map_new_cp.pkl")

    parser.add_argument("--train_percentage", help="train data split", type=int, default=Macros.default_train_percentage)
    parser.add_argument("--validation_percentage", help="validation data split", type=int, default=Macros.default_validation_percentage)
    parser.add_argument("--test_percentage", help="test data split", type=int, default=Macros.default_test_percentage)
    # parser.add_argument("--model", type=str, choices=["codebert", "codet5"], default="codebert")
    # parser.add_argument("--is_cp", help="whether cross project or not", action="store_true")

    args = parser.parse_args()
    # if args.is_cp:
    #     print("NOTE: you must set mutants_file and test_file manually when using this option")
    utils.set_seed(Macros.random_seed)
    main(args)
