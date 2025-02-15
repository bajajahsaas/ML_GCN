#!/bin/bash
#
#SBATCH --job-name=GCN_coco
#SBATCH --output=logscoco/hh_%j.txt  # output file
#SBATCH -e logscoco/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u  demo_coco_gcn.py data/coco --image-size 448 --batch-size 8 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
#sleep 1
exit

