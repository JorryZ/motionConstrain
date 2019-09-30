# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:26:00 2019

@author: JorryZ

File: __init__.py
Description: load motionConstrain and motionConstrainSolver function

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V3.1.0 release version
                                                        -motionConstrain version 3.0.0
                                                        -motionConstrainSolver version 3.1.0
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V3.1.1 release version
                                                        -motionConstrain version 3.0.0
                                                        -motionConstrainSolver version 3.1.1                                                        
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V3.1.2 release version
                                                        -motionConstrain version 3.0.0
                                                        -motionConstrainSolver version 3.1.2                                               

Requirements:
    numpy
    math
    scipy
    motionSegmentation
    medImgProc
    trimesh
    json (optional)
    pickle (optional)
All rights reserved.
"""
_version='3.1.2'
print('motionConstrain version',_version)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from motionConstrain import *
from motionConstrainSolver import *
