# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:45:42 2019

@author: bieZY (e0348827@u.nus.edu)
Method: velocityWise, displacementWise
Modules: numpy, imageio, medImgProc, motionSegmentation

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: jorry.zhengyu@gmail.com         15NOV2019           -V5.0.0 release version, process with BSF model file
          function: pointTrace, coefCombiner, coefZeroRemap, timeRemap, reShape,vtk2img, lazySnapImg, divFree, meshVolume, errorCalc, imgScaling, vtkSampling
  Author: jorry.zhengyu@gmail.com         18Dec2019           -V5.0.2 release version, import issue

"""

import numpy as np
import os
import sys
import imageio
import medImgProc as medImgProc
import medImgProc.image as image

import motionSegmentation.BsplineFourier as BsplineFourier
import motionSegmentation.bfSolver as bfSolver
import motionConstrain.motionConstrain as motionConstrain

print('postProcessBSF version 5.0.2')
print('postProcessBSF Help: input funcName(funcHelp=True) to get help for functions')

# edit part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pointTrace(BSFfile=None, STLfile=None, savePath=None, customPath=None, funcHelp=False):
    if funcHelp:
        print('Function pointTrace is to track the motion of points from STL file, and generate new STL over cycle.')
        print('Inputs of this function are: path+name of BSFfile, path+name of STLfile, name of saving folder (savePath).')
        print('A sub-folder is automatically created to save files, folder name: stl+customPath')
        return
    try:
        savePath=savePath+'\\stl-'+customPath
    except:
        savePath=savePath+'\\stl'
    solver=bfSolver.bfSolver()
    solver.bsFourier=BsplineFourier.BsplineFourier(coefFile=BSFfile)
    solver.pointTrace(stlFile=STLfile, savePath=savePath, timeList=int(solver.bsFourier.spacing[3]))
    print('function pointTrace done! Have a happy day ^_^')
    
def timeRemap(rawBSFfile=None, newBSFfile=None, time=0, funcHelp=False):
    if funcHelp:
        print('Function timeRemap is to remap BSF to appointed time point as new reference timing.')
        print('Inputs of this function are: path+name of rawBSFfile, path+name of newBSFfile, time as reference timing (default 0).')
        return
    solver=bfSolver.bfSolver()
    solver.bsFourier=BsplineFourier.BsplineFourier(rawBSFfile)
    #fourierTerms=solver.bsFourier.coef.shape[3]//2
    solver.initialize()
    solver.pointsCoef=[]
    for m in range(len(solver.points)):
        coef=solver.bsFourier.getRefCoef(solver.points[m])
        solver.pointsCoef.append(coef.copy())
    solver.bsFourier.regridToTime(solver.points,solver.pointsCoef,time=time)
    solver.bsFourier.writeCoef(newBSFfile)
    
def coefCombiner(rawBSFfile=None, newBSFfilePath=None, newBSFfile=None, time=0, funcHelp=False):
    if funcHelp:
        print('Function coefCombiner is to combine all results of different fourier terms.')
        print('Inputs of this function are: path+name of rawBSFfile, saving folder path of newBSFfile (newBSFfilePath), name of newBSFfile time as reference timing (default 0).')
        return
    saveFtermPath=newBSFfilePath+'\\coefFT'
    solver=motionConstrain.motionConstrain()
    solver.initialize(coefFile=rawBSFfile)
    fourierTerms=solver.coefMat.shape[3]//2
    filepath=[]
    for fterm in range(1,(fourierTerms*2+1)):
        filepath.append(saveFtermPath+str(fterm)+'.txt')
    solver.coefCombiner(filepath)
    solver.coefZeroRemap(time=time)
    solver.writeFile((newBSFfilePath+'\\'+newBSFfile),coefMat=1)
    
def coefZeroRemap(rawBSFfile=None, newBSFfile=None, time=0, funcHelp=False):
    if funcHelp:
        print('Function coefZeroRemap is to calculate the value of DC term (zero term) after processing the Fourier terms.')
        print('Inputs of this function are: path+name of rawBSFfile, path+name of newBSFfile, time as reference timing (default 0).')
        return
    solver=motionConstrain.motionConstrain()
    solver.initialize(coefFile=rawBSFfile)
    solver.coefZeroRemap(time=time)
    solver.writeFile(newBSFfile,coefMat=1)
    
def reShape(rawBSFfile=None, newBSFfile=None, finalShape=None, funcHelp=False):
    if funcHelp:
        print('Function reShape is to change the size of the bsplineFourier matrix (size of control point matrix).')
        print('Inputs of this function are: path+name of rawBSFfile, path+name of newBSFfile, desired shape (finalShape: [x,y,z,ft,uvw]).')
        return
    solver=bfSolver.bfSolver()
    solver.bsFourier=BsplineFourier.BsplineFourier(rawBSFfile)
    try:
        solver.bsFourier=solver.bsFourier.reshape(finalShape)
        solver.bsFourier.writeCoef(newBSFfile)
    except:
        print('Error: please input correct finalShape [x,y,z,ft,uvw]')
    
def vtk2img(VTKfile=None, savePath=None, funcHelp=False):
    if funcHelp:
        print('Function vtk2img is to convert 3D VTK to 2D slice image (png).')
        print('Inputs of this function are: path+name of VTKfile, saving folder path of image (savePath).')
        return
    os.makedirs(savePath, exist_ok=True)
    img=medImgProc.imread(VTKfile)
    img.imwrite2D(savePath,axes=['y','x'])
#    vtkPath='D:\\4.Cardiac motion tracking\\v{0:d}\\3DUS\\VTK\\'.format(case)
#    timeNum=timeNumDict[case]
#    #timeNum=17
#    for i in range(timeNum):
#        vtkName=vtkPath+'VTK{:0>2d}'.format(i)
#        img=medImgProc.imread(vtkName+'.vtk')
#        img.imwrite2D(vtkName,axes=['y','x'])
    
def lazySnapImg(VTKfile=None, savePath=None, funcHelp=False):
    if funcHelp:
        print('Function lazySnapImg is to convert 3D VTK to 2D slice image (png in lazySnap format).')
        print('Inputs of this function are: path+name of VTKfile, saving folder path of image (savePath).')
        return
    os.makedirs(savePath, exist_ok=True)
    img=medImgProc.imread(VTKfile)
    currentDim=img.dim[:]
    axes=['z','y','x']
    transposeIndex,currentDim=image.arrangeDim(currentDim,axes,False)
    data=img.data.transpose(transposeIndex)
    zSlice=data.shape[0]
    for m in range(zSlice):
        imageio.imwrite(os.path.normpath(savePath+'/slice{:0>3d}time{:s}'.format(m+1,savePath[-3:])+'.png'),data[m])
#    vtkPath='D:\\4.Cardiac motion tracking\\v{0:d}\\3DUS\\VTK\\'.format(case)
#    #timeNum=timeNumDict[case]
#    timeNum=1
#    for n in range(timeNum):  #time points for lazySnap, time+1
#        vtkName=vtkPath+'VTK{:0>2d}'.format(n)
#        savePath=vtkPath+'/time{:0>3d}'.format(n+1)
#        os.makedirs(savePath, exist_ok=True)
#        img=medImgProc.imread(vtkName+'.vtk')
#        currentDim=img.dim[:]
#        axes=['z','y','x']
#        transposeIndex,currentDim=image.arrangeDim(currentDim,axes,False)
#        data=img.data.transpose(transposeIndex)
#        zSlice=data.shape[0]
#        for m in range(zSlice):
#            imageio.imwrite(os.path.normpath(savePath+'/slice{:0>3d}time{:0>3d}'.format(m+1,n+1)+'.png'),data[m])
    
def meshVolumeError(meshVolumeFile=None, funcHelp=False):
    if funcHelp:
        print('Function meshVolumeError is to calculate RMS error of mesh volume over cylce.')
        print('Inputs of this function are: path+name of meshVolumeFile.')
        return
    meshVolume=np.loadtxt(meshVolumeFile,delimiter=' ')
    totalVolume=np.sum(meshVolume,axis=0)
    print('totalVolume: ',totalVolume)
    rmsError=0
    for i in range(len(meshVolume)):
        volume=meshVolume[i,:]
        #meanV=np.mean(volume)
        squaredError=0.0
        for j in range(1,len(volume)):
            squaredError+=(volume[j]-volume[0])**2
        rmsError+=np.sqrt(squaredError/(len(volume)-1))
    print('mesh volume rms error: ',rmsError,' ; divided by mesh number: ', rmsError/len(meshVolume))
    return (rmsError, totalVolume)
    
def divFree(funcHelp=False):
    if funcHelp:
        print('Function divFree is to calculate divergence value.')
        print('unfinish~~~~~~')
    
def imgScaling(funcHelp=False):
    if funcHelp:
        print('Function imgScaling is to scale image.')
        print('unfinish~~~~~~')
    
def vtkSampling(funcHelp=False):
    if funcHelp:
        print('Function vtkSampling is to sample points from 3D VTK.')
        print('unfinish~~~~~~')

