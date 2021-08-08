#!/bin/bash
#SBATCH -p gpu20
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --time 12:00:00
