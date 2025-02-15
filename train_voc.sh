#!/bin/bash
#
#SBATCH --job-name=GCN_voc
#SBATCH --output=logsvoc/hh_%j.txt  # output file
#SBATCH -e logsvoc/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 8 -e --resume checkpoint/voc/voc_checkpoint.pth.tar
#sleep 1
exit

