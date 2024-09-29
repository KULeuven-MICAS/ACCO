# ACCO: Automated Causal CNN Scheduling Optimizer for Real-Time Edge Accelerators
## Paper:
[Yin J, Mei L, Guntoro A, et al. "ACCO: Automated Causal CNN Scheduling Optimizer for Real-Time Edge Accelerators." 2023 IEEE 41st International Conference on Computer Design (ICCD). IEEE, 2023.](https://arxiv.org/abs/2406.07161)


## Abstract
ACCO is an automated optimizer that explores efficient Causal CNN transformation and DF scheduling for ST-CNNs on edge hardware accelerators. It is built upon our previous depth-first DNN scheduling DSE framework [DeFiNES, HPCA, 2023](https://arxiv.org/abs/2212.05344). 

This repository contains the basic scripts to perform the cost-model-based DSE and the result analysis for the experiments in the paper. For more low-level implementation details and explanations, please refer to the [DeFiNES repository](https://github.com/KULeuven-MICAS/DeFiNES).

## Installation

1) Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) environment

2) Download ACCO (`git clone` this repo)

3) Use a terminal or an Anaconda Prompt for the following steps:
   -  `cd` into the ACCO repo
   -  Create the conda environment from the environment.yml file
       ```
       conda env create -f environment.yml
       ```
   -  Activate the new environment:
       ```
       conda activate ACCOenv
       ```

## Run

The exploration includes the following steps. Please refer to each scripts' comments for further explanation and available tweaks.

### Step 1 
Run
```
python DfStackCutAndXform_mp_misc.py
```

#### What does this script do?
It triggers the main DSE execution across the combination of assigned hardware model, DNN workload model, tiling options, depth-first fusion ranges, and etc. The output of this script is the cost model ensemble of each scenario stored in .pkl files.

Please note that if using a broad range of workloads and DSE options in one launch, the execution time and disk usage would significantly grow.

### Step 2
Run
```
python explore_pickle_compile_readings.py
```

#### What does this script do?
It parses the cost model ensembles from the previous step's output and compiles reports in terms of the frame-wise, layer-wise, and total hardware cost of different depth-first scheduling flavours. These reports are on the level of each fused depth-first layer trunk, so that this is the preparation step for the final exploration of the best strategy across the entire DNN workload.

### Step 3
Run
```
python explore_pickle_hierarchy_cache_allocation.py
```

#### What does this script do?
This script leverages the distilled fuse-layer-level cost model reports to find the best overall scheduling strategy across the entire DNN workload.

---

## Citation 

```
@inproceedings{yin2023acco,
  title={ACCO: Automated Causal CNN Scheduling Optimizer for Real-Time Edge Accelerators},
  author={Yin, Jun and Mei, Linyan and Guntoro, Andre and Verhelst, Marian},
  booktitle={2023 IEEE 41st International Conference on Computer Design (ICCD)},
  pages={391--398},
  year={2023},
  organization={IEEE}
}
```

We are continually improving the framework, and welcoming all questions and feedback. 

We hope our tool can help other researchers to better explore and understand the vast DNN accelerator architecture-and-scheduling design space and can offer the best design solutions.

For the latest features of this framework, please refer to the main repository [ZigZag](https://github.com/KULeuven-MICAS/zigzag) and [Stream](https://github.com/KULeuven-MICAS/stream).







