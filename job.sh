#!/bin/bash

dataset_list=("3A4" "CB1" "DPP4" "HIVINT" "HIVPROT" "LOGD" "METAB" "NK1" "OX1" "OX2" "PGP" "PPB" "RAT_F" "TDI" "THROMBIN")

for dataset in "3A4"; do
    for size in 1.0; do
        for seed in {1..50}; do
            echo "running $dataset $size $seed"
            python3 sheridan-vs-conformal.py $dataset $size $seed &
        done
    done
done

