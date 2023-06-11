#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --partition=csmpi_fpga_short
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --time=0:05:00
#SBATCH --output=opencl.out

export XILINX_XRT=/opt/xilinx/xrt

#Compile on the machine, not the head node
make bin/stencil_cl

bin/stencil_cl 125 100 >> results/stencil.txt
bin/stencil_cl 125 100 >> results/stencil.txt
bin/stencil_cl 125 100 >> results/stencil.txt
bin/stencil_cl 125 100 >> results/stencil.txt
bin/stencil_cl 125 100 >> results/stencil.txt
bin/stencil_cl 125 1000 >> results/stencil.txt
bin/stencil_cl 125 1000 >> results/stencil.txt
bin/stencil_cl 125 1000 >> results/stencil.txt
bin/stencil_cl 125 1000 >> results/stencil.txt
bin/stencil_cl 125 1000 >> results/stencil.txt
bin/stencil_cl 125 10000 >> results/stencil.txt
bin/stencil_cl 125 10000 >> results/stencil.txt
bin/stencil_cl 125 10000 >> results/stencil.txt
bin/stencil_cl 125 10000 >> results/stencil.txt
bin/stencil_cl 125 10000 >> results/stencil.txt

