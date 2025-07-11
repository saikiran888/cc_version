#!/bin/bash

# Parse command-line arguments
while getopts h:a:b:d:s:e:n:m:o:x:c:d flag
do
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
    esac
done

# Set CSV output file (Excel can open CSV files)
excel_file="${res_dir}/${dataid}/timing_comp_results.csv"

# Record overall start time
total_start=$(date +%s)

########################################
# 1. Initiate Results
########################################
initiate_start=$(date +%s)
/home/sagar/.conda/envs/DT-pipeline-cpu/bin/python initiate_results.py $home $res_dir $dataid $sblk $eblk $numframes $seg $change_channels $cc_threshold
initiate_end=$(date +%s)
time_initiate=$((initiate_end - initiate_start))
echo "Completed initiate_results in ${time_initiate} seconds."

########################################
# 2. Process Block Part 1
########################################
p1_start=$(date +%s)
for ((i=sblk; i<=eblk; i=$(( i+1 )) ));
do   
     echo "starting part 1 blocks $i"
#     /home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_block_part1.py $home $raw_dir $res_dir $dataid $i $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk
     echo "completed part 1 blocks $i"
done
p1_end=$(date +%s)
time_part1=$((p1_end - p1_start))
echo "Completed process_block_part1 in ${time_part1} seconds."

########################################
# 3. Process Block Part 2
########################################
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
#           /home/eddie/.conda/envs/gpu/bin/python process_block_part2.py $home $raw_dir $res_dir $dataid $cur_ind $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk $j &
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

########################################
# 4. Process Block Part 3
########################################
p3_start=$(date +%s)
for ((i=sblk; i<=eblk; i=$(( i+1 )) ));
do
   echo "starting part 3 blocks $i"
#   /home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_block_part3.py $home $raw_dir $res_dir $dataid $i $numframes $seg $overwrite $change_channels $cc_threshold $sblk $eblk
   echo "completed part 3 blocks $i"
done
p3_end=$(date +%s)
time_part3=$((p3_end - p3_start))
echo "Completed process_block_part3 in ${time_part3} seconds."

########################################
# 5. Process process_result.py
########################################
result_start=$(date +%s)
#/home/sagar/.conda/envs/DT-pipeline-cpu/bin/python process_result.py $home $raw_dir $res_dir $dataid $sblk $eblk $numframes $seg $change_channels $cc_threshold 
result_end=$(date +%s)
time_result=$((result_end - result_start))
echo "Completed process_result.py in ${time_result} seconds."

########################################
# (Optional) ET-ratio.py call
########################################
#/home/sagar/.conda/envs/DT-pipeline-cpu/bin/python ET-ratio.py $home $dataid $sblk $eblk $res_dir $numframes $seg $change_channels $cc_threshold 

########################################
# 6. Process process_secondary_analysis.py
########################################
secondary_start=$(date +%s)
/home/eddie/.conda/envs/gpu/bin/python process_secondary_analysis.py $res_dir $dataid 
secondary_end=$(date +%s)
time_secondary=$((secondary_end - secondary_start))
echo "Completed process_secondary_analysis.py in ${time_secondary} seconds."

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
  echo "Step,TimeTaken(s)"
  echo "Initiate Results,${time_initiate}"
  echo "CROPPING,${time_part1}"
  echo "CELL DETECTION,${time_part2}"
  echo "TRACKING,${time_part3}"
  echo "FEATURE EXTRACTION,${time_result}"
  echo "process_secondary_analysis.py,${time_secondary}"
  echo "Total Time,${total_time}"
} >> "$excel_file"

echo "Timing details written to ${excel_file}"
