# dino_lowrank

## Introduction
Experimenting with DINO low-rank pretraining for developing a general echo image/patch-level embedding

## Installation
### Docker
- Initialize a docker container running pytorch at https://hub.docker.com/r/pytorch/pytorch/tags. The code is ran using pytorch:2.0.1-cuda11.7-cudnn8-runtime
- Pytorch version may be changed depending on which CUDA version is on your local device
- refer to docker_setup for example
- Run setup.sh within the container to install relevant libraries

### Conda
- Configure a conda environment with a gpu-compatible pytorch version
- Install the list of files in setup.sh

## Preparing data
Important: Please DATASET_ROOT argument on line 15 of dataloader_tmed.py to where your dataset is located.
- The folder should be organized as follows for TMED within DATASET_ROOT
  - DEV[165/479/56] folders
  - dataset folders
  - csv files

## Usage
Training using low-rank DINO
- Refer to dino_experiments_launch.sh for an example of training command
- Check which GPUs are available using the nvidia-smi command and specify which GPUs are visible to the script ahead of time.
  - Typically >1 GPU is need for DINO training.
  - Set --nproc_per_node equal to the number of visible GPUs.
  - Pending: script for running job on SLURM cluster.
- Refer to main_dino.py for what the arugments do
  - Set the --output_dir directory to somewhere you know
 
Evaluation
- Refer to the respective Eval_*.ipynb for a range of options for evaluating the embeddings from DINO
  
Checklist before your first run
- Did you install relevant libraries in setup.py?
- Did you put the data in the relevant folder?
- Did you read dino_experiments_launch.sh and edit relevant fields?
- Did you reserve the GPUs you run on using clustermarket?
- Do you understand where the output files from your training will go?
