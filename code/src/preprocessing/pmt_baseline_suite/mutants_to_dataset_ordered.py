import argparse
import itertools
import os, sys
import random
from threading import current_thread
import numpy as np
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from sklearn.preprocessing import LabelBinarizer
from preprocessing.pmt_baseline.CodePreprocessor import CodePreprocessor
from Macros import Macros
from preprocessing import utils
import pickle

def build_mutators(src_dir, mutators):
    for filepath in tqdm(os.listdir(src_dir)):
        with open(f"{src_dir}/{filepath}","rb") as sf:
            suites = pickle.load(sf)
        
        for suite in suites: 
            for mutant in suite["mutants"]:
                mutators.add(mutant["mutator"])

def map_src_to_dst(src_dir, dest_dir, mutators, is_train=False):
    vocab_method_name = set()
    vocab_body = set()

    for filepath in tqdm(os.listdir(src_dir)):
        with open(f"{src_dir}/{filepath}","rb") as sf:
            mutants = pickle.load(sf)
        
        with open(f"{dest_dir}/{filepath}", "wb") as df:
            mapped_mutants = mutants_to_dataset(mutants, mutators, vocab_method_name, vocab_body)
            pickle.dump(mapped_mutants, df)

    if is_train:
        vocab_method_name.remove("<num>")
        vocab_method_name = ["<pad>", "<unk>", "<num>"] + sorted(list(vocab_method_name))

        vocab_body = ["<pad>", "<unk>"] + sorted(list(vocab_body))

        with open(Macros.defects4j_root_dir / dest_dir / ".." / "vocab_method_name.pkl", "wb") as f:
            pickle.dump(vocab_method_name, f) 

        with open(Macros.defects4j_root_dir / dest_dir / ".." / "vocab_body.pkl", "wb") as f:
            pickle.dump(vocab_body, f) 

def mutants_to_dataset(mutants, mutators, vocab_method_name, vocab_body):
    stop_words_for_method = ["test"]
    stop_words_for_body = ["#", "\\\\", "\\", "\\n"]

    body_cp = CodePreprocessor(stop_words_for_body, remove_num=False)
    method_cp = CodePreprocessor(stop_words_for_method)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(mutators)

    new_mutants = []
    for suite in mutants:
        suite_new = {"label": suite["label"], "mutants": []}
        for mutant in suite["mutants"]:
            method_tokens = method_cp.run(mutant["source_method"])
            line_tokens = body_cp.run(mutant["src_lines"][mutant["mut_src_line_no"]])
            before_tokens = body_cp.run(mutant["before_pmt"])
            after_tokens = body_cp.run(mutant["after_pmt"])

            test_tokens = method_cp.run(mutant["test_method"])
            vocab_method_name.update(method_tokens)
            vocab_method_name.update(test_tokens)
            vocab_body.update(line_tokens)
            vocab_body.update(before_tokens)
            vocab_body.update(after_tokens)
            suite_new["mutants"].append({"mut_no": mutant["mut_no"], "source_method": method_tokens, "line": line_tokens, "mutator": label_binarizer.transform([mutant["mutator"]])[0], "before": before_tokens, "after": after_tokens, "test_method": test_tokens, "label": mutant["label"]})
        new_mutants.append(suite_new)

    return new_mutants

def main(args):
    (Macros.defects4j_root_dir / args.dst_set / "train").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / args.dst_set / "val").mkdir(exist_ok=True, parents=True)
    (Macros.defects4j_root_dir / args.dst_set / "test").mkdir(exist_ok=True, parents=True)

    mutators = set()
    build_mutators((Macros.defects4j_root_dir / args.src_set / "train"), mutators)
    build_mutators((Macros.defects4j_root_dir / args.src_set / "val"), mutators)
    build_mutators((Macros.defects4j_root_dir / args.src_set / "test"), mutators)

    mutators = list(mutators)
    map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "train"), (Macros.defects4j_root_dir / args.dst_set / "train"), mutators, is_train=True)
    map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "val"), (Macros.defects4j_root_dir / args.dst_set / "val"), mutators)
    map_src_to_dst((Macros.defects4j_root_dir / args.src_set / "test"), (Macros.defects4j_root_dir / args.dst_set / "test"), mutators)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_set", help="source set", default="base_set_suite")
    parser.add_argument("--dst_set", help="dest set", default="pmt_baseline_ordered_suite")

    args = parser.parse_args()
    utils.set_seed(Macros.random_seed)
    main(args)
