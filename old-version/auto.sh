#!/bin/bash

bsub -n 1 -q k40 -m mic02 -o job.out ./pme --xyz data/N1000_Phi0.1.xyz --dim 128 --porder 6
