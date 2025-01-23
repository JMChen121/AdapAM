# <font size=6>AdapAM</font>
This repository contains the source code for the paper: **Adversarial Attack on Black-Box Multi-Agent by Adaptive Perturbation**.

# Overview

We propose **AdapAM**, a novel learning-based framework for **Adap**tive adversarial **A**ttacks on the black-box **M**AS. 

The overview of **AdapAM** is shown in the figure below:
![图片](images/overview.png)

# Environment Setup
Our experiments are conducted on three popular multi-agent benchmarks with different characteristics as follows.

## Required Environment 

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [Multiagent Particle-World Environments (MPE)](https://github.com/openai/multiagent-particle-envs)
- [Google Research Football (GRF)](https://github.com/google-research/football)

## Installation instructions
Install Python packages

```shell
# require Anaconda 3 or Miniconda 3
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
pip install protobuf==3.19.5 sacred==0.8.2 numpy scipy gym==0.11 matplotlib \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger

# Even though we provide requirement.txt, it may have redundancy. We recommend to install other required packages by running the code and finding which required package hasn't installed yet.

```

Set up SMAC:

```shell
bash install_sc2.sh

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.
```

Set up GRF:

```shell
bash install_gfootball.sh
```

Set up MPE:

```shell
# install this package first
pip install seaborn
```

# Running
## Train the target agents
```
### SMAC:

## SMAC-1c3s5z
bash scripts/train/train_smac_1c3s5z.sh

## SMAC-8m
bash scripts/train/train_smac_8m.sh

####################################

### GRF:

## GRF-counter_attack
bash scripts/train/train_grf_counter_attack.sh

## GRF-3vs1_with_keeper
bash scripts/train/train_grf_3vs1_with_keeper.sh

####################################

### MPE:

## MPE-spread
bash scripts/train/train_mpe_spread.sh

## MPE-reference
bash scripts/train/train_mpe_reference.sh
```

## Train the **AdapAM**
```
### SMAC:

## SMAC-1c3s5z
bash scripts/train/train_smac_1c3s5z_AdapAM.sh

## SMAC-8m
bash scripts/train/train_smac_8m_AdapAM.sh

####################################

### GRF:

## GRF-counter_attack
bash scripts/train/train_grf_counter_attack_AdapAM.sh

## GRF-3vs1_with_keeper
bash scripts/train/train_grf_3vs1_with_keeper_AdapAM.sh

####################################

### MPE:

## MPE-spread
bash scripts/train/train_mpe_spread_AdapAM.sh

## MPE-reference
bash scripts/train/train_mpe_reference_AdapAM.sh
```

## Evaluation
```
### SMAC:

## SMAC-1c3s5z
bash scripts/eval/eval_smac_1c3s5z.sh

## SMAC-8m
bash scripts/eval/eval_smac_8m.sh

####################################

### GRF:

## GRF-counter_attack
bash scripts/eval/eval_grf_counter_attack.sh

## GRF-3vs1_with_keeper
bash scripts/eval/eval_grf_3vs1_with_keeper.sh

####################################

### MPE:

## MPE-spread
bash scripts/eval/eval_mpe_spread.sh

## MPE-reference
bash scripts/eval/eval_mpe_reference.sh
```


# Reference
- https://github.com/hijkzzz/pymarl2/
- https://github.com/marlbenchmark/on-policy/