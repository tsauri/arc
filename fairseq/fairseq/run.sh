#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=24:00:00
source /etc/profile.d/modules.sh
source /home/acb11328ra/.bashrc
module load cuda/9.2/9.2.88.1
# source /home/acb11328ra/anaconda3/etc/profile.d/conda.sh
# export PATH="/home/acb11328ra/anaconda3/bin:$PATH"

$@
