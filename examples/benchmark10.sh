# example script to run multiple examples, the code and results will be in:
base='benchmark_multiple_networks'
codefolder=$base/code
resultsfolder=$base/results
lofgile=$base/log.txt

# Exit at first failing example run
#set -e -o pipefail

mkdir -p $base
for m in 5 10 50; do
    for duration in 0.1 1 10; do
        for sp_flag in "" "no-"; do
            for prm_flag in "" "no-"; do
                for cond_flag in "" "no-"; do
                    args="--devicename cuda_standalone \
                        --resultsfolder $resultsfolder \
                        --codefolder $codefolder \
                        --M $m \
                        --profiling \
                        --monitors \
                        --duration $duration \
                        --"$mt_flag"multi_threading" \
                        --"$cond_flag"use_conditional_connect" \
                        --"$prm_flag"PRM"

                    cmd="python brunelhakim10.py --no-heterog-delays $args"
                    echo $cmd
                    $cmd 2>&1 | tee -a $logfile
                done
            done
        done
    done
done
