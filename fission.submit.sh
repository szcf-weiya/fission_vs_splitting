#!/bin/bash

ncpu=25
for sigma in 0 0.1; do
    for rho in $(seq 0 0.1 0.9); do
        for delta in $(seq 0.1 0.1 0.6); do
            sbatch -c $ncpu --export=rho=$rho,ncpu=$ncpu,sigma=$sigma,delta=$delta fission.job
    	done
    done
done
