#!/bin/bash
# The next line will skip to the binary data
SKIP=`awk '/^__ARCHIVE_BELOW__/ { print NR + 1; exit 0; }' $0`
# Extract the binary data and decode it
tail -n+$SKIP $0 | base64 -d > PWMLFF.tar.gz
# Now you can extract the tar file and do your stuff
BASE_DIR=$(pwd)
TEMP_DIR=$(mktemp -d)
FINAL_DIR=$BASE_DIR/PWMLFF-March2024   # change the PWMLFF directory name
ENV_DIR=$FINAL_DIR/env  # change the conda environment directory name
mkdir -p $FINAL_DIR
mkdir -p $ENV_DIR
pushd $TEMP_DIR
# extract the all the files
tar -xvzf $BASE_DIR/PWMLFF.tar.gz     # this tar file contains the conda environment and the PWMLFF directory
tar -xvzf pwmlff_env.tar.gz -C $ENV_DIR    # extract the conda environment
popd
mv $TEMP_DIR/PWMLFF $FINAL_DIR  
source $ENV_DIR/bin/activate
cd $FINAL_DIR/PWMLFF/src
sh build.sh
rm -rf $TEMP_DIR
exit 0
# Adjust the above line, and then append the base64 encoded tar file
__ARCHIVE_BELOW__