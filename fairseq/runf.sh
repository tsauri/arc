#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
. /etc/profile.d/modules.sh
. /home/9/17R70036/.bashrc 

#module load cuda/9.0.176  \
#module load cudnn/7.3  \
$@
# --continue-from wsj_models/deepspeech_45.pth.tar
