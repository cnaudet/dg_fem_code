#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -N gpu_practice

module load cuda
./test/mesh/test_msh.cpp

