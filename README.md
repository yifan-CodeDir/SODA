# SODA
----------
Welcome to the website of SODA. This repository contains a replication package for our research paper named **"Spotting Code Mutation for Predictive Mutation Testing"** accepted by [ASE 2024](https://conf.researchr.org/home/ase-2024). We provide our code, data and results for the ease of replicating our experiments.

- **Code** : We provide all the code needed to replicate our experiment in the `code` subdirectory. We also provide the guidance to run our code in the following page.
- **Data** : We provide the data in `code/data.zip`. We also provide the raw logs produced by Major in `raw_data`.
- **Result**: We provide all our results in the `results` subdirectory. In particular, we provide results for all the three models (Seshat, MutationBERT, SODA) in cross-version and cross-project scenarios with two granularities (test matrix and test suite).
- **Additional Result**: We provide some additional results in markdown files in the `results` subdirectory, e.g., confirmation check, some mispredicted examples, evaluation on newer versions, F-beta scores, Venn diagrams, etc.

----------
## Environment Preparation
- `cd code`
- `conda env create -f requirements.yaml`
- `conda activate soda`
----------
## Data Preparation
We provide the data in `code/data.zip`, you can directly unzip it and use it (but still need a few further processing steps detailed in the following page) for training or evaluation process.
- `cd code`
- `unzip data.zip`
  
For ones who are interested in richer information, we also provide the raw logs in `raw_data/major_raw_logs.tar.gz`. This compressed file contains the raw logs produced by Major including `covMap.csv`, `killMap.csv`, `mutants.log`, `testMap.csv`. We provide the raw logs not only for the versions we used in Defects4J, but also for some newer versions whose details are given in `results/newer_version.md`.

----------
## Preprocessing the data and training models
## Test matrix-level
### For Seshat
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_seshat_cross_project.py` ; output: seshat_cross_project_base_set_foldx
  - 3. `python3 ./pmt_baseline/mutants_to_dataset_ordered.py --src_set ../../data/seshat_cross_project_base_set_foldx --dst_set ../../data/seshat_cross_project_pmt_baseline_foldx` ; output: seshat_cross_project_pmt_baseline_foldx (The foldx could be from "fold1" to "fold5")
  - ##### Training model
  - 1. `cd code/src/runtime/pmt_baseline_cross_project`
  - 2. `bash run.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_seshat_cross_version.py` ; output: seshat_cross_version_base_set
  - 3. `python3 ./pmt_baseline/mutants_to_dataset_ordered_cross_version.py` ; output: seshat_cross_version_pmt_baseline_ordered
  - ##### Training model
  - 1. `cd code/src/runtime/pmt_baseline_cross_version`
  - 2. `bash run.sh`

### For MutationBERT
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_cross_project_with_hier_sample_pos.py` ; output: cross_project_base_set_foldx
  - 3. `python3 ./codebert_token_diff/mutants_to_dataset.py --src_set ../../cross_project_base_set_foldx --dst_set ../../cross_project_codebert_token_diff_foldx` ; output: cross_project_codebert_token_diff_foldx
  - ##### Training model
  - 1. `cd code/src/runtime/trans_codebert_cross_project`
  - 2. `bash run.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_cross_version_with_hier_sample_pos.py` ; output: cross_version_base_set
  - 3. `python3 ./codebert_token_diff/mutants_to_dataset_cross_version.py` ; output: cross_version_codebert_token_diff 
  - ##### Training model
  - 1. `cd code/src/runtime/trans_codebert_cross_version`
  - 2. `bash run.sh`

### For SODA
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_cross_project_with_hier_sample_pos.py` ; output: cross_project_base_set_foldx
  - 3. `python3 ./codet5_token_diff/mutants_to_dataset.py --src_set ../../cross_project_base_set_foldx --dst_set ../../cross_project_soda_foldx` ; output: cross_project_soda_foldx
  - ##### Training model
  - 1. `cd code/src/runtime/soda_cross_project`
  - 2. `bash run.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_cross_version_with_hier_sample_pos.py` ; output: cross_version_base_set
  - 3. `python3 ./codet5_token_diff/mutants_to_dataset_cross_version.py` ; output: cross_version_soda
  - ##### Training model
  - 1. `cd code/src/runtime/soda_cross_version`
  - 2. `bash run.sh`




## Test suite-level
### For Seshat
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_project.py` ; output: cross_project_suite_base_set_foldx
  - 3. `python3 ./pmt_baseline_suite/mutants_to_dataset_ordered.py --src_set cross_project_suite_base_set_foldx --dst_set cross_project_suite_pmt_baseline_ordered_foldx` ; output: cross_project_suite_pmt_baseline_ordered_foldx
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cp`
  - 2. `bash run_pmt.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_version.py` ; output: cross_version_suite_base_set
  - 3. `python3 ./pmt_baseline_suite/mutants_to_dataset_ordered_cross_version.py` ; output: cross_version_suite_pmt_baseline_ordered
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cv`
  - 2. `bash run_pmt.sh`

### For MutationBERT
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_project.py` ; output: cross_project_suite_base_set_foldx
  - 3. `python3 ./codebert_token_diff_suite/mutants_to_dataset.py --src_set cross_project_suite_base_set_foldx --dst_set cross_project_suite_codebert_token_diff_foldx` ; output: cross_project_suite_codebert_token_diff_foldx
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cp`
  - 2. `bash run_codebert.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_version.py` ; output: cross_version_suite_base_set
  - 3. `python3 ./codebert_token_diff_suite/mutants_to_dataset_cross_version.py` ; output: cross_version_suite_codebert_token_diff 
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cv`
  - 2. `bash run_codebert.sh`

### For SODA
- #### Cross-project
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_project.py` ; output: cross_project_suite_base_set_foldx
  - 3. `python3 ./codet5_token_diff_suite/mutants_to_dataset.py --src_set cross_project_suite_base_set_foldx --dst_set cross_project_suite_soda_foldx` ; output: cross_project_suite_soda_foldx
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cp`
  - 2. `bash run_soda.sh`

- #### Cross-version
  - ##### Preprocessing
  - 1. `cd code/src/preprocessing`
  - 2. `python3 build_mutant_set_suite_cross_version.py` ; output: cross_version_suite_base_set
  - 3. `python3 ./codet5_token_diff_suite/mutants_to_dataset_cross_version.py` ; output: cross_version_suite_soda
  - ##### Find the best threshold
  - 1. `cd code/src/runtime/suite_cv`
  - 2. `bash run_soda.sh`
----------
## Evaluation
- `cd src/evaluation`
- To run test matrix-level prediction: `python3 eval_models_cross_project.py` or `python3 eval_models_cross_version.py`
- To run test suite-level prediction: `python3 eval_suite_cross_project.py` or `python3 eval_suite_cross_version.py`
