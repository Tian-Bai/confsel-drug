#!/bin/bash

dataset_list=("3A4" "CB1" "DPP4" "HIVINT" "HIVPROT" "LOGD" "METAB" "NK1" "OX1" "OX2" "PGP" "PPB" "RAT_F" "TDI" "THROMBIN")

for dataset in "${dataset_list[@]}"; do
    for size in 0.1; do
        for seed in {1..100}; do
            echo "running $dataset $size $seed"
            nohup python sheridan-vs-conformal.py $dataset $size $seed &
        done
    done
done

wait

