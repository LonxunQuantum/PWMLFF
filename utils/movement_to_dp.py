#!/usr/bin/env python
import dpdata
import os
import sys
import argparse

# init all args
#fname_movement = 'MOVEMENT'
#num_samples_per_set = 2000

# parsing args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='MOVEMENT0')
parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=2000)
parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='data0')
args = parser.parse_args()

#dpdata.LabeledSystem(args.input, fmt='pwmat/movement').to('deepmd/raw', args.directory)
dpdata.LabeledSystem(args.input, fmt='MOVEMENT').to('deepmd/raw', args.directory)
#os.system('cd ' + args.directory +'; raw_to_set.sh ' + str(args.number) + '; cd ../')
