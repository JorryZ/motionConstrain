# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:45:42 2019

@author: JorryZ
Objective: apply incompressibility constraint to b-Spline Fourier model using divergence free
Method: velocityWise, displacementWise
Modules: numpy, scipy, imageio, matplotlib, trimesh, simpleitk, motionSegmentation, motionConstrain

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V2.3 old general version
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V3.0.0 release version, general version
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V3.1.0 release version, make motionConstrainSolver a module
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V3.1.1 optional for json
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V3.1.2 import error correction
  Author: jorry.zhengyu@gmail.com         30SEPT2019           -V3.1.3 pointSampling error correction
  Author: jorry.zhengyu@gmail.com         01Oct2019            -V3.2.0 pointSampling error correction
"""
print('motionConstrainSolver version 3.2.0')
print('Warning: the bsFourier.txt should be in the real time, not in the phantom time, like "f3_t1".')

import os
import sys
import numpy as np
import json
import trimesh
import medImgProc
import motionSegmentation.BsplineFourier as BsplineFourier
import motionSegmentation.bfSolver as bfSolver
import motionConstrain.motionConstrain as motionConstrain

class mcSolver:
    # Needed files: bsf.txt, gt.vtk and img, input.stl
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self):
        self.motionConstrain=motionConstrain.motionConstrain()
        self.casePath=None
        self.bsfName=None
        self.sampleSource=None                  #sample points from: stl mask (stl), vtk mask (vtk), global mask (gm)
        self.stlName=None                       #'time1_TOF-fine.stl'
        self.vtkName=None                       #'gt.vtk'
        self.dimlen=None                        #{'x':0.1676,'y':0.1676,'z':1.2794}
        
        self.spacingDivision=None
        self.bgGlobalField=None                 # True means the background part includes sample part, if False add extra in the end of folder
        self.finalShape=None
        self.sizeSingleMat=None
        self.reShape=None
        
    def initialize(self,casePath=None,bsfName=None,finalShape=None,gap=1,SampleCoords=True):
        self.casePath=casePath
        self.bsfName=bsfName
        
        self.motionConstrain.initialize(coefFile=(casePath+'\\'+bsfName+'.txt'),spacingDivision=[1.,1.5],gap=0,SampleCoords=True)
        rawShape=self.motionConstrain.coefMat.shape
        if type(finalShape)==type(None) or finalShape==rawShape:
            finalShape=rawShape
            reShape=False   #False: none; True: type2 reShape
        else:
            reShape=True
        sizeSingleMat=np.prod(finalShape[:3])
        self.finalShape=finalShape
        self.reShape=reShape
        self.sizeSingleMat=sizeSingleMat
        
    def pointSampling(self,sampleSource=None,stlName=None,vtkName=None,dimlen=None,sampleRatio=[0.3,0.4,3.,3.5],spacingDivision=[5.,1.5],bgGlobalField=True):
        self.dimlen=dimlen
        self.sampleSource=sampleSource
        self.bgGlobalField=bgGlobalField
        print('bgGlobalField: ',bgGlobalField)
        
        if sampleSource=='stl':
            #sample points from stl file
            try:
                sampleSourcePath=self.casePath+stlName
            except:
                print('Warning: source of sample part from STL file, please specify a STL file!')
                sys.exit()
            sampleData=trimesh.load(sampleSourcePath)
            skip=4
            self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,skip=skip)
            if type(sampleRatio)!=type(None):
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat>sampleRatio[3]):
                    spacingDivision[1]-=0.1
                    self.motionConstrain.bgCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=0))
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat<sampleRatio[2]):
                    spacingDivision[1]+=0.1
                    self.motionConstrain.bgCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=0))
                while(len(self.motionConstrain.motionConstrain.sampleCoord)/len(self.motionConstrain.bgCoord)>sampleRatio[1]):
                    skip+=1
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,skip=skip)
                while(len(self.motionConstrain.sampleCoord)/len(self.motionConstrain.bgCoord)<sampleRatio[0]):
                    skip-=1
                    if skip==0:
                        print('Please make a finer STL mesh!!!')
                        sys.exit()
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,skip=skip)
            print('sampleSource: stl mask!! bgCoord spacingDivision: ',spacingDivision[1])
        
        elif sampleSource=='vtk':
            try:
                sampleSourcePath=self.casePath+'\\'+vtkName
                sampleData=medImgProc.imread(sampleSourcePath)
                sampleData.dim=['z','y','x']
                sampleData.dimlen=dimlen
                sampleData.rearrangeDim(['x','y','z'])
                print('Dimlen: ',sampleData.dimlen)
            except:
                print('Warning: source of sample part from VTL file, please specify a VTK file and a dimlen (dimension length)!')
                sys.exit()
            
            self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,spacingDivision=spacingDivision,bgGlobalField=bgGlobalField)
            if type(sampleRatio)!=type(None):
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat>sampleRatio[3]):
                    spacingDivision[1]-=0.1
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,spacingDivision=spacingDivision,bgGlobalField=bgGlobalField)
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat<sampleRatio[2]):
                    spacingDivision[1]+=0.1
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,spacingDivision=spacingDivision,bgGlobalField=bgGlobalField)
                while(len(self.motionConstrain.sampleCoord)/len(self.motionConstrain.bgCoord)>sampleRatio[1]):
                    spacingDivision[0]-=0.2
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,spacingDivision=spacingDivision,bgGlobalField=bgGlobalField)
                while(len(self.motionConstrain.sampleCoord)/len(self.motionConstrain.bgCoord)<sampleRatio[0]):
                    spacingDivision[0]+=0.2
                    self.motionConstrain.samplePointsFromSource(sampleSource=sampleSource,sampleData=sampleData,spacingDivision=spacingDivision,bgGlobalField=bgGlobalField)
            print('sampleSource: vtk mask!! spacingDivision: ',spacingDivision)
        
        elif sampleSource=='gm':
            if type(sampleRatio)!=type(None):
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat>sampleRatio[3]):
                    spacingDivision[1]-=0.1
                    self.motionConstrain.bgCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=0))
                while(len(self.motionConstrain.bgCoord)/self.sizeSingleMat<sampleRatio[2]):
                    spacingDivision[1]+=0.1
                    self.motionConstrain.bgCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=0))
                spacingDivision[0]=spacingDivision[1]
                self.motionConstrain.sampleCoord=self.motionConstrain.samplePoints(spacingDivision=spacingDivision[0],gap=0)
                while(len(self.motionConstrain.sampleCoord)/len(self.motionConstrain.bgCoord)<sampleRatio[1]):
                    spacingDivision[0]+=0.1
                    self.motionConstrain.sampleCoord=self.motionConstrain.samplePoints(spacingDivision=spacingDivision[0],gap=0)
                '''
                if type(customSpacingDivision)!=type(None):
                    spacingDivision[0]=customSpacingDivision
                    solver.sampleCoord=solver.samplePoints(spacingDivision=spacingDivision[0],gap=0)
                else:
                    spacingDivision[0]=spacingDivision[1]
                    solver.sampleCoord=solver.samplePoints(spacingDivision=spacingDivision[0],gap=0)
                    while(len(solver.sampleCoord)/len(solver.bgCoord)<sampleRatio[1]):
                        spacingDivision[0]+=0.1
                        solver.sampleCoord=solver.samplePoints(spacingDivision=spacingDivision[0],gap=0)
                '''
            else:
                self.motionConstrain.sampleCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[0],gap=0))
                self.motionConstrain.bgCoord=np.array(self.motionConstrain.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=0))
            print('sampleSource: global mask!! spacingDivision: !!',spacingDivision)
        self.spacingDivision=spacingDivision
        print('Sampling done: sampleCoord has %d points, bgCoord has %d points, '%(len(self.motionConstrain.sampleCoord),len(self.motionConstrain.bgCoord)))
        
    def solve(self,mode='displacementWise',weight=[7.,1.],fterm_start=1,customPath=None):
        weight=np.sqrt(weight)
        if self.reShape==True:
            sampleCoord=self.motionConstrain.sampleCoord.copy()
            bgCoord=self.motionConstrain.bgCoord.copy()
            
            self.motionConstrain=motionConstrain.motionConstrain()
            try:
                self.motionConstrain.initialize(coefFile=(self.casePath+'\\'+self.bsfName+'_{0:d}-{1:d}-{2:d}.txt'.format(self.finalShape[0],self.finalShape[1],self.finalShape[2])),SampleCoords=True,gap=5)
            except:
                solver2=bfSolver.bfSolver()
                solver2.bsFourier=BsplineFourier.BsplineFourier(self.casePath+'\\'+self.bsfName+'.txt')#+'_f'+str(fourierTerms)+'.txt')
                try:
                    #finalShape.append(solver2.bsFourier.coef.shape[3])
                    #finalShape.append(solver2.bsFourier.coef.shape[4])
                    print('reshape in process ',self.finalShape,'......')
                    solver2.bsFourier=solver2.bsFourier.reshape(self.finalShape)
                    solver2.bsFourier.writeCoef(self.casePath+'\\'+self.bsfName+'_{0:d}-{1:d}-{2:d}.txt'.format(self.finalShape[0],self.finalShape[1],self.finalShape[2]))
                except:
                    print('Error: please input correct finalShape [x,y,z,ft,uvw]')
                    sys.exit()
                del solver2
                self.motionConstrain.initialize(coefFile=(self.casePath+'\\'+self.bsfName+'_{0:d}-{1:d}-{2:d}.txt'.format(self.finalShape[0],self.finalShape[1],self.finalShape[2])),SampleCoords=True,gap=5)
                print('reshape done: ',self.finalShape,'^_^_^_^_^_^')
                #sys.exit()
            self.motionConstrain.sampleCoord=sampleCoord.copy()
            self.motionConstrain.bgCoord=bgCoord.copy()
        
        if self.bgGlobalField==True:
            savePath=self.casePath+'\\'+'{0:s}_{1:.1f}-{2:.1f}_w{3:.1f}-{4:.1f}_{5:d}-{6:d}-{7:d}_{8:s}'.format(mode,self.spacingDivision[0],self.spacingDivision[1],weight[0],weight[1],self.finalShape[0],self.finalShape[1],self.finalShape[2],self.sampleSource)
        else:
            savePath=self.casePath+'\\'+'{0:s}_{1:.1f}-{2:.1f}_w{3:.1f}-{4:.1f}_{5:d}-{6:d}-{7:d}_{8:s}_extra'.format(mode,self.spacingDivision[0],self.spacingDivision[1],weight[0],weight[1],self.finalShape[0],self.finalShape[1],self.finalShape[2],self.sampleSource)
        try:
            savePath=savePath+'_'+customPath+'\\'
            print('CustomPath setting: ',customPath)
        except:
            savePath=savePath+'\\'
        os.makedirs(savePath, exist_ok=True)
        print('Processing with savePath: ',savePath,'~~~~~~~~~~~~')
        
        saveFtermPath=savePath+'coefFT'
        np.savetxt((savePath+'\\'+self.bsfName+'_sampleCoord'+'.txt'),self.motionConstrain.sampleCoord,fmt='%.8f',delimiter=' ')
        np.savetxt((savePath+'\\'+self.bsfName+'_bgCoord'+'.txt'),self.motionConstrain.bgCoord,fmt='%.8f',delimiter=' ')
        
        self.motionConstrain.getMatB()
        self.motionConstrain.getMatdBdX()
        self.motionConstrain.getMatdUdX()
        # errorCalc test
        self.motionConstrain.errorCalc(savePath=savePath,option='Init')   # divergence error of points
        self.motionConstrain.solve(method=mode,maxIteration=100,maxError=0.01,fterm_start=fterm_start,saveFtermPath=saveFtermPath,weight=weight,convergence=0.5,reportevery=60)
        self.motionConstrain.errorCalc(savePath=savePath,option='Final')
        self.motionConstrain.coefZeroRemap(remap=0)
        self.motionConstrain.writeFile((savePath+'\\'+self.bsfName+'_coefMat'+'.txt'),coefMat=1)
        self.motionConstrain.writeFile((savePath+'\\'+self.bsfName+'_dCoefMat'+'.txt'),dCoefMat=1)
        np.savetxt((savePath+'\\'+self.bsfName+'_rmsList'+'.txt'),self.motionConstrain.rmsList,fmt='%.8f',delimiter=' ')
        try:
            if type(self.motionConstrain.dimlen)!=type(None):
                with open((savePath+'\\'+'scale_xyz.txt'), 'w') as f:
                    json.dump(self.motionConstrain.dimlen, f)
        except:
            pass
        print('One case done, savepath:  ',savePath,'  *_*_*_*_*_*')
