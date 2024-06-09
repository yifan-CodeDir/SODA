import os
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

"""
Class containing all global constants for the project
"""
class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    src_dir: Path = this_dir
    project_dir: Path = src_dir.parent

    data_dir: Path = project_dir / "data"
    results_dir: Path = project_dir / "_results"
    defects4j_root_dir: Path = data_dir
    defects4j_data_dir: Path = data_dir / "raw_data"
    defects4j_pmt_save_dir: Path = data_dir / "pmt_parsed"
    defects4j_trans_save_dir: Path = data_dir / "trans_parsed"

    downloads_dir: Path = project_dir / "_downloads"
    model_dir: Path = project_dir / "models"
    defects4j_model_dir: Path = model_dir
    model_configs_dir: Path = project_dir / "src" / "model_configs"
    log_dir: Path = src_dir / "runtime" / "logs"
    eval_dir: Path = src_dir / "evaluation"

    latest_versions: dict = {"Lang": 1, "Chart": 1, "Gson": 15, "Cli": 30, "JacksonCore": 25, "Csv": 15}
    # for cross version evaluation
    all_versions: dict = {"Lang": [60,50,40,30,20,10,1], "Chart": [25,20,15,10,5,1], "Gson": [1,5,10,15], "Cli": [1,10,20,30], "JacksonCore": [1,5,10,15,20,25], "Csv": [1,5,10,15]}
    train_versions: dict = {"Lang": [60,50,40,30,20], "Chart": [25,20,15,10], "Gson": [1,5], "Cli": [1,10], "JacksonCore": [1,5,10,15], "Csv": [1,5]}
    # train_versions: dict = {"Cli": [1,10]}
    valid_versions: dict = {"Lang": [10], "Chart": [5], "Gson": [10], "Cli": [20], "JacksonCore": [20], "Csv": [10]}
    # valid_versions: dict = {"Cli": [20]}
    test_versions: dict = {"Lang": [1], "Chart": [1], "Gson": [15], "Cli": [30], "JacksonCore": [25], "Csv": [15]}
    # test_versions: dict = {"Cli": [30]}

    default_train_percentage: int = 80
    default_validation_percentage: int = 10
    default_test_percentage: int = 10
    
    max_mutator_size = 20
    max_method_name = 20

    class_weights = [1, 6.14] # 86% of data is 0, 14% is 1 (0.86/0.14 = 6.14)
    class_weights_diff = [1, 2.57] # 72% of data is 0, 28% is 1 (0.72/0.28 = 2.57)
    class_weights_suite = [1.56, 1] # 39% of data is 0, 61% is 1 (0.61/0.39 = 1.56)

    class_weights_diff_cross_version = {"Chart":[1, 3.76], "Cli":[1, 2.57], "Csv":[1, 1.44], "Gson":[1, 1.78], "JacksonCore":[1, 2.70], "Lang":[1, 1.78]}
    random_seed = 10

    NUM_MUTATORS = 11

    MODEL_DICT = {"codebert": {"tokenizer": AutoTokenizer, "model": AutoModel, "max_embedding_size": 512, "pretrained":"microsoft/codebert-base"},
                  "codet5": {"tokenizer": AutoTokenizer, "model": AutoModel, "max_embedding_size": 512, "pretrained":"salesforce/codet5-base"}
                  }
