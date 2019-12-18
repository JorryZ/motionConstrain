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
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V3.1.3 release version
                                                        -motionConstrain version 3.0.0
                                                        -motionConstrainSolver version 3.1.3 
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V3.2.0 release version
                                                        -motionConstrain version 3.0.0
                                                        -motionConstrainSolver version 3.2.0                                                   
  Author: jorry.zhengyu@gmail.com         01Oct2019              -VT4.0.0 test version
                                                        -motionConstrain version T4.0.0
                                                        -motionConstrainSolver version 3.2.0                                                        
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V4.1.0 test version
                                                        -motionConstrain test version 4.1.0
                                                        -motionConstrainSolver version 3.2.1 
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V4.1.1 test version
                                                        -motionConstrain test version 4.1.1
                                                        -motionConstrainSolver version 3.2.1 
  Author: jorry.zhengyu@gmail.com         02Oct2019              -V4.2.0 test version
                                                        -motionConstrain test version 4.2.0
                                                        -motionConstrainSolver version 3.2.3                                                         
  Author: jorry.zhengyu@gmail.com         03Oct2019              -V4.2.1 test version
                                                        -motionConstrain test version 4.2.1
                                                        -motionConstrainSolver version 3.3.0
  Author: jorry.zhengyu@gmail.com         03Oct2019              -V4.2.2 test version
                                                        -motionConstrain test version 4.2.2
                                                        -motionConstrainSolver version 3.3.1 
  Author: jorry.zhengyu@gmail.com         31Oct2019              -V5.0.0 release version
                                                        -motionConstrain release version 5.0.0
                                                        -motionConstrainSolver release version 5.0.0
                                                        -postProcessBSF release version 5.0.0
  Author: jorry.zhengyu@gmail.com         19NOV2019              -V5.0.1 release version
                                                        -motionConstrain release version 5.0.1
                                                        -motionConstrainSolver release version 5.0.1
                                                        -postProcessBSF release version 5.0.0   
  Author: jorry.zhengyu@gmail.com         18Dec2019              -V5.0.2 release version
                                                        -motionConstrain release version 5.0.1
                                                        -motionConstrainSolver release version 5.0.1
                                                        -postProcessBSF release version 5.0.2                           
Requirements:
    numpy
    scipy
    motionSegmentation
    medImgProc
    trimesh
    json (optional)
    pickle (optional)
All rights reserved.
"""
_version='5.0.2'
print('motionConstrain version',_version)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import motionConstrain.motionConstrain as motionConstrain
import motionConstrain.motionConstrainSolver as motionConstrainSolver
