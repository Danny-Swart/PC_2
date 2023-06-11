#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --partition=csmpi_fpga_short
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --time=0:05:00
#SBATCH --output=seq.out

export XILINX_XRT=/opt/xilinx/xrt

#Compile on the machine, not the head node
make bin/stencil_seq

bin/stencil_seq 10 100 > results/stencil_seq.txt
