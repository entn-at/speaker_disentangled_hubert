#!/bin/bash

#$ -cwd                      ## Execute a job in the current directory
#$ -l node_f=2               ## Use number of node
#$ -l h_rt=24:00:00          ## Running job time
#$ -j y                      ## Integrate standard error output into a standard output
#$ -p -5
#$ -m abe
#$ -M EMAIL_ADDRESS

config=${1:-configs/speechlm/default.yaml}
config_file=${2:-configs/speechlm/default_config.yaml}

module load openmpi/5.0.7-nvhpc
module load cudnn/9.0.0
module load nccl/2.20.5
module load miniconda

main_process_ip=$(head -n 1 $PE_HOSTFILE | awk '{print $1}')

mpirun \
    -npernode 1 \
    -n 2 \
    -x LD_LIBRARY_PATH \
    bash -c '
    eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
    conda activate t4
    accelerate launch \
        --config_file='"${config_file}"' \
        --main_process_ip='"${main_process_ip}"' \
        --machine_rank=${OMPI_COMM_WORLD_RANK} \
        main_speechlm.py train \
        --config='"${config}"'
'