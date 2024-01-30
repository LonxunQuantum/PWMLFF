#!/usr/bin/env python
import json
import os, sys
from src.pwdata.pwdata import Save_Data
from src.user.extract_raw import Extract_Param

if __name__ == "__main__":
    json_path = sys.argv[1]
    # json_path = "/data/home/hfhuang/2_MLFF/2-DP/19-json-version/4-CH4-dbg/extract.json"
    os.chdir(os.path.dirname(os.path.abspath(json_path)))
    json_file = json.load(open(json_path))
        
    params = Extract_Param(json_file)
    dir = params.get_data_file_dict()
    raw_data_path = params.file_paths.raw_path
    datasets_path = os.path.join(params.file_paths.json_dir, dir["trainSetDir"])
    train_data_path = dir["trainDataPath"] 
    valid_data_path = dir["validDataPath"]
    train_ratio = params.train_valid_ratio
    data_shuffle = params.valid_shuffle
    seed = params.seed
    format = params.format
    for data_path in raw_data_path:
        Save_Data(data_path, datasets_path, train_data_path, valid_data_path, 
                  train_ratio, data_shuffle, seed, format)