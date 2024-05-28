#!bin/bash
# This script will zip all the files and conda environment for the MLFF model
# run this script in the directory include the PWMLFF
# every step can be run in the terminal separately!

# pack the conda environment
# with this command, the conda environment will be packed in the file pwmlff.tar.gz
# before running this command, make sure you have activated the conda environment
conda pack -n pwmlff_env

# pack the files: PWMLFF is the directory
# with this command, the conda environment and the files will be packed in the file PWMLFF.tar.gz
tar -czf PWMLFF.tar.gz pwmlff_env.tar.gz PWMLFF

# base64 the tar.gz file
base64 PWMLFF.tar.gz > PWMLFF.tar.gz.base64

# Adjust the all_unzip.sh file, and then append the base64 encoded tar file: copy following command, and run it in the terminal
cat PWMLFF.tar.gz.base64 >> all_unzip.sh
