#!/bin/bash

# Slurm parameters
MEMO=25G                     # Memory required (50G)
TIME=3-00:00:00              # Time required (3d)
CORE=10                      # Cores required (10)

# Assemble order prefix
ORDP="sbatch --account=def-yyang --mem="$MEMO" -n 1 -c "$CORE" --time="$TIME" --chdir=/home/tianbai/scratch/confsel-drug"

# Create directory for log files
LOGS=logs
mkdir -p $LOGS

dataset_list=("3A4" "CB1" "DPP4" "HIVINT" "HIVPROT" "LOGD" "METAB" "NK1" "OX1" "OX2" "PGP" "PPB" "RAT_F" "TDI" "THROMBIN")

for dataset in "${dataset_list[@]}"; do
    for size in 1.0; do
        for seed in {1..100}; do
            # Assemble slurm order for this job
            JOBN=$dataset"_"$size"_"$seed

            # Submit the job
            SCRIPT="submit-3z.sh $dataset $size $seed"
                
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"

            # Assemble slurm order for this job
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            # ORD="scancel -n "$JOBN  

            # Print order
            echo $ORD

            # Submit order
            $ORD
        done
    done
done

