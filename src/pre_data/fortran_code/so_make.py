#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
import platform

if platform.system()!='Linux':
    raise NotImplementedError("This script just can run in Linux os!")

pythonVersion=platform.python_version()

if pythonVersion[0]!='3':
    raise NotImplementedError("This script just should run in python3!")

verStr=platform.python_implementation().lower()+'-'+pythonVersion[0]+pythonVersion[2]+'m'
machineStr=platform.machine()
osStr=verStr+'-'+machineStr
oldOsStr='cpython-35m-x86_64'
command='sed s/'+oldOsStr+'/'+osStr+'/g makefile_bak >makefile'
os.system(command)
command='make py'
os.system(command)
