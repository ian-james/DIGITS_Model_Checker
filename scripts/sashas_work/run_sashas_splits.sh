#!/bin/bash

for num in {49..53}; do
  python ./src/split_ethans_csv.py \
    -f ~/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_${num}/DIGITS_C_${num}_DIGITS_Upload_Output_Trials_1-5.csv \
    -o ~/Projects/Western/Western_Postdoc/Analysis/test/C_${num}/
done
