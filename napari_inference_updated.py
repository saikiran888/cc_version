#!/bin/bash

# Initialize pipeline control variables to 'no'
run_initiate=no
run_part1=no
run_part2=no
run_part3=no
run_result=no
run_et_ratio=no
run_secondary=no

# Parse command-line arguments
echo "Received arguments: $@"
while getopts h:a:b:d:s:e:n:m:o:x:c:i:j:k:l:r:q:v:z flag
do
    echo "DEBUG: Processing flag: ${flag} with value: ${OPTARG}"
    case "${flag}" in
        h) home=${OPTARG};;
        a) raw_dir=${OPTARG};;
        b) res_dir=${OPTARG};;
        d) dataid=${OPTARG};;
        s) sblk=${OPTARG};;
        e) eblk=${OPTARG};;
        n) numframes=${OPTARG};;
        m) seg=${OPTARG};;
        o) overwrite=${OPTARG};;
        x) change_channels=${OPTARG};;
        c) cc_threshold=${OPTARG};;
        i) run_initiate=${OPTARG};;
        j) run_part1=${OPTARG};;
        k) run_part2=${OPTARG};;
        l) run_part3=${OPTARG};;
        r) run_result=${OPTARG};;
        q) run_et_ratio=${OPTARG};;
        v) run_secondary=${OPTARG};;
        z) dummy=${OPTARG};;
    esac
done

echo "Parsed arguments:"
echo "home=$home"
echo "raw_dir=$raw_dir"
echo "res_dir=$res_dir"
echo "dataid=$dataid"
echo "sblk=$sblk"
echo "eblk=$eblk"
echo "numframes=$numframes"
echo "seg=$seg"
echo "overwrite=$overwrite"
echo "change_channels=$change_channels"
echo "cc_threshold=$cc_threshold"
echo "run_initiate=$run_initiate"
echo "run_part1=$run_part1"
echo "run_part2=$run_part2"
echo "run_part3=$run_part3"
echo "run_result=$run_result"
echo "run_et_ratio=$run_et_ratio"
echo "run_secondary=$run_secondary"
echo "DEBUG: Checking if run_secondary is empty: '$run_secondary'"

# Set CSV output file (Excel can open CSV files)
excel_file="${res_dir}/${dataid}/timing_comp_results.csv"

# Record overall start time
total_start=$(date +%s)

########################################
# 1. Initiate Results
########################################
if [ "$run_initiate" = "yes" ]; then
    initiate_start=$(date +%s)
    echo "Calling initiate_results.py with arguments: $home $res_dir $dataid $sblk $eblk $numframes $seg $change_channels $cc_threshold"
    /home/sagar/.conda/envs/DT-pipeline-cpu/bin/python initiate_results.py $home $res_dir $dataid $sblk $eblk $numframes $seg $change_channels $cc_threshold 2>/dev/null
    initiate_end=$(date +%s)
    time_initiate=$((initiate_end - initiate_start))
    echo "Completed initiate_results in ${time_initiate} seconds."
else
    echo "Skipping initiate_results (disabled)"
    time_initiate=0
fi

########################################
# 2. Process Block Part 1
########################################
if [ "$run_part1" = "yes" ]; then
    p1_start=$(date +%s)
    for ((i=sblk; i<=eblk; i=$(( i+1 )) ));
    do   
         echo "starting part 1 blocks $i"
    #     /home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_block_part1.py $home $raw_dir $res_dir $dataid $i $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk 2>/dev/null
         echo "completed part 1 blocks $i"
    done
    p1_end=$(date +%s)
    time_part1=$((p1_end - p1_start))
    echo "Completed process_block_part1 in ${time_part1} seconds."
else
    echo "Skipping process_block_part1 (disabled)"
    time_part1=0
fi

########################################
# 3. Process Block Part 2
########################################
if [ "$run_part2" = "yes" ]; then
    p2_start=$(date +%s)
    declare -a pids
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    for ((i=sblk; i<=eblk; i=$(( i+num_gpus )) ));
    do
        for ((j=0; j<num_gpus; j=$(( j+1 )) ));
        do
            cur_ind=$((i+j))
            if (( cur_ind <= eblk )); then
                echo "starting part 2 blocks $cur_ind"
    #           /home/eddie/.conda/envs/gpu/bin/python process_block_part2.py $home $raw_dir $res_dir $dataid $cur_ind $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk $j & 2>/dev/null
                pids+=($!)
            fi
        done

        wait

        for pid in "${pids[@]}"; do
            kill "$pid" 2>/dev/null
        done
        pids=()
    done
    p2_end=$(date +%s)
    time_part2=$((p2_end - p2_start))
    echo "Completed process_block_part2 in ${time_part2} seconds."
else
    echo "Skipping process_block_part2 (disabled)"
    time_part2=0
fi

########################################
# 4. Process Block Part 3
########################################
if [ "$run_part3" = "yes" ]; then
    p3_start=$(date +%s)
    for ((i=sblk; i<=eblk; i=$(( i+1 )) ));
    do
       echo "starting part 3 blocks $i"
    #   /home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_block_part3.py $home $raw_dir $res_dir $dataid $i $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk 2>/dev/null
       echo "completed part 3 blocks $i"
    done
    p3_end=$(date +%s)
    time_part3=$((p3_end - p3_start))
    echo "Completed process_block_part3 in ${time_part3} seconds."
else
    echo "Skipping process_block_part3 (disabled)"
    time_part3=0
fi

########################################
# 5. Process process_result.py
########################################
if [ "$run_result" = "yes" ]; then
    result_start=$(date +%s)
    #/home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_result.py $home $raw_dir $res_dir $dataid $sblk $eblk $numframes $seg $change_channels $cc_threshold 2>/dev/null
    result_end=$(date +%s)
    time_result=$((result_end - result_start))
    echo "Completed process_result.py in ${time_result} seconds."
else
    echo "Skipping process_result.py (disabled)"
    time_result=0
fi

########################################
# (Optional) ET-ratio.py call
########################################
if [ "$run_et_ratio" = "yes" ]; then
    et_start=$(date +%s)
    #/home/sagar/.conda/envs/DT-pipeline-cpu/bin/python ET-ratio.py $home $dataid $sblk $eblk $res_dir $numframes $seg $change_channels $cc_threshold 2>/dev/null
    et_end=$(date +%s)
    time_et=$((et_end - et_start))
    echo "Completed ET-ratio.py in ${time_et} seconds."
else
    echo "Skipping ET-ratio.py (disabled)"
    time_et=0
fi

########################################
# 6. Process process_secondary_analysis.py
########################################
if [ "$run_secondary" = "yes" ]; then
    secondary_start=$(date +%s)
    /home/eddie/.conda/envs/gpu/bin/python process_secondary_analysis.py $res_dir $dataid 2>/dev/null
    secondary_end=$(date +%s)
    time_secondary=$((secondary_end - secondary_start))
    echo "Completed process_secondary_analysis.py in ${time_secondary} seconds."
else
    echo "Skipping process_secondary_analysis.py (disabled)"
    time_secondary=0
fi

########################################
# Total elapsed time
########################################
total_end=$(date +%s)
total_time=$((total_end - total_start))
echo "Total Elapsed Time: ${total_time} seconds"

########################################
# Write timing details to CSV file with BOM for UTF-8
########################################
# Prepend the UTF-8 BOM so Excel interprets the file correctly
printf '\xEF\xBB\xBF' > "$excel_file"
{
  echo "Step,TimeTaken(s),Status"
  echo "Initiate Results,${time_initiate},$([ "$run_initiate" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "CROPPING,${time_part1},$([ "$run_part1" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "CELL DETECTION,${time_part2},$([ "$run_part2" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "TRACKING,${time_part3},$([ "$run_part3" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "FEATURE EXTRACTION,${time_result},$([ "$run_result" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "ET-ratio,${time_et},$([ "$run_et_ratio" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "process_secondary_analysis.py,${time_secondary},$([ "$run_secondary" = "yes" ] && echo "Executed" || echo "Skipped")"
  echo "Total Time,${total_time},Completed"
} >> "$excel_file"

echo "Timing details written to ${excel_file}"
