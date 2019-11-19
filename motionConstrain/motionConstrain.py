# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:26:00 2019

@author: JorryZ
Method: velocityWise, displacementWise
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: jorry.zhengyu@gmail.com         01MARCH2019           -V1.0 Created, motionConstrain.velocityWise
  Author: jorry.zhengyu@gmail.com         01APRIL2019           -V2.0 Add motionConstrain.displacementWise
  Author: jorry.zhengyu@gmail.com         04MAY2019             -V2.1 Add Weight
  Author: jorry.zhengyu@gmail.com         12JUNE2019             -V2.2 Add motionConstrain.samplePoints function
  Author: jorry.zhengyu@gmail.com         12JUNE2019             -V2.3 Add motionConstrain.errorCalc function
  Author: jorry.zhengyu@gmail.com         12JUNE2019             -V2.4 Add normalization weight in motionConstrain.solve_displacementWise, UNDO
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V2.5 Add weight in solve_velocityWise
  
  Author: jorry.zhengyu@gmail.com         30SEPT2019             -V3.0.0 release version, modify func motionConstrain.samplePointsFromFile and change name to samplePointsFromSource
  
  Author: jorry.zhengyu@gmail.com         01Oct2019              -VT4.0.0 test version, use elements to calculate error (original use sum), maxError from 0.0001 to 0.00001
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V4.1.0 test version, add weight for Buvw
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V4.1.1 test version, code check
  Author: jorry.zhengyu@gmail.com         01Oct2019              -V4.1.2 test version, func errorCalc modify
  Author: jorry.zhengyu@gmail.com         02Oct2019              -V4.2.0 test version, add regular for ftCoefuvw (norm, fast)
  Author: jorry.zhengyu@gmail.com         03Oct2019              -V4.2.1 test version, solve_displacementWise, set certain coef to 0
  Author: jorry.zhengyu@gmail.com         04Oct2019              -V4.2.2 test version, deltarms calculation
  Author: jorry.zhengyu@gmail.com         10Oct2019              -V4.3.0 test version, calculate deltaC, normalize RMSMat by period and point number by 1/spacingDivision
  Author: jorry.zhengyu@gmail.com         30Oct2019              -V5.0.0 release version, sampling normalization, two regular for ICP (norm, greed)
  Author: jorry.zhengyu@gmail.com         19NOV2019              -V5.0.1 release version, fix up error for regular
"""
print('motionConstrain version 5.0.1')
import sys
import numpy as np
import math
import scipy as sp
from scipy.sparse.linalg import splu
import motionSegmentation.BsplineFourier as BsplineFourier

class motionConstrain:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self):
        '''
        Initialize motionConstrain class
        '''
        self.time=None
        self.origin=None
        self.spacing=None   #grid spacing (x,y,z)
        self.period=None
        self.coefMat=[]
        self.shape=None
        self.coordMat=None
        
        self.dimlen=None    #physical length of each (x,y,z) pixel
        self.sampleCoord =None
        self.bgCoord=None  #imgReg BsplineFourier result
        self.spacingDivision= None
        
        self.bsFourier=BsplineFourier.BsplineFourier()
        self.BMat=[]    #[point,n]
        self.dBdXMat=[]
        self.dUdXMat=[] # [point, TF, uvw]
        
        self.method=None
        self.dCoefMat=[]   # [point,FT,uvw]
        self.savePath=None
        self.rmsList=[]
        self.errorList=None     # divergence error of points
        print('motionConstrain_init_ done')
        
    def initialize(self,coefFile=None,origin=None,shape=None,spacing=None,period=None,time=None,fourierFormat=None,fileScale=1.,delimiter=' ',SampleCoords=False,spacingDivision=[1.,1.5],gap=0):
        
        if type(coefFile)!=type(None):
            self.bsFourier.readFile(coefFile=coefFile,origin=origin,shape=shape,spacing=spacing,fourierFormat=fourierFormat,delimiter=delimiter)
            self.origin=self.bsFourier.origin
            self.spacing=self.bsFourier.spacing[:3]
            self.period=self.bsFourier.spacing[3]
            self.coefMat=self.bsFourier.coef.copy()
            self.shape=self.coefMat.shape
            self.coordMat = np.mgrid[self.origin[0]:(self.origin[0]+(self.shape[0]-0.1)*self.spacing[0]):self.spacing[0], self.origin[1]:(self.origin[1]+(self.shape[1]-0.1)*self.spacing[1]):self.spacing[1],self.origin[2]:(self.origin[2]+(self.shape[2]-0.1)*self.spacing[2]):self.spacing[2]].reshape(3,*self.shape[:3]).transpose(1,2,3,0)
            #self.coefMat=self.bsFourier.coef.reshape(-1, order='F')
        if SampleCoords==True:
            self.sampleCoord=np.array(self.bsFourier.samplePoints(spacingDivision=spacingDivision[0],gap=gap))  #self.bsFourier.samplePoints()
            self.bgCoord=np.array(self.bsFourier.samplePoints(spacingDivision=spacingDivision[1],gap=gap))
            print('Initialization: sampleCoord has %d points, bgCoord has %d points.'%(len(self.sampleCoord),len(self.bgCoord)))
        
        if type(time)!=type(None):
            self.time=time
            print('Initialization: time=',self.time)
        
        if type(self.bsFourier.coef)==np.ndarray:
          print('Initialization: shape=',self.bsFourier.coef.shape)
          print('Initialization: spacing=',self.bsFourier.spacing)
          print('Initialization: origin=',self.bsFourier.origin)
        
    '''    
    def secondSamplePoints(self):
        try:
            pass
        except:
            pass
    '''
    
    def samplePointsFromSource(self,sampleSource=None,sampleData=None,scale=1.,spacingDivision=[5.,1.5],bgGlobalField=False,skip=None):
        '''
        bgGlobalField: the field of background points, True: include muscle part, False: not include muscle part
        skip: for stl file to select points
        '''
        if sampleSource=='stl':
            stlData=np.array(sampleData.vertices)/scale
            if type(skip)!=type(None):
                temp=[]
                for i in range(len(stlData)):
                    if i%skip==0:
                        temp.append(stlData[i])
            else:
                temp=stlData
            self.sampleCoord=np.array(temp.copy())
            print('sample coordinates from STL file,  have ',len(self.sampleCoord),' points')

        elif sampleSource=='vtk':
            vtkData=sampleData
            self.dimlen=vtkData.dimlen
            
            samplepts1=self.bsFourier.samplePoints(spacingDivision=spacingDivision[0])
            wall=[]
            for m in range(len(samplepts1)):
                if vtkData.getData(samplepts1[m])>=0.5:
                    wall.append(samplepts1[m].copy())
            self.sampleCoord=wall.copy()
            self.sampleCoordSize=len(self.sampleCoord)
            
            samplepts2=self.bsFourier.samplePoints(spacingDivision=spacingDivision[1])
            if bgGlobalField==False:
                bg=[]
                for m in range(len(samplepts2)):
                    if vtkData.getData(samplepts2[m])<0.5:
                        bg.append(samplepts2[m].copy())
                self.bgCoord=bg.copy()
            else:
                self.bgCoord=samplepts2.copy()
            del vtkData
            print('Sample coordinates {0:d} points, background coordinates {1:d} points'.format(len(self.sampleCoord),len(self.bgCoord)))
        
    def samplePoints(self,spacingDivision=2.,gap=0):
        '''
        sample grid points inside the image field
        '''
        print('motionConstrain.samplePoints running: sample grid points inside the image field ^_^')
        gridx0=int(math.ceil(abs(self.origin[0]/self.spacing[0])))
        gridx1=int(math.ceil(abs(self.origin[0]*2/self.spacing[0])))
        gridy0=int(math.ceil(abs(self.origin[1]/self.spacing[1])))
        gridy1=int(math.ceil(abs(self.origin[1]*2/self.spacing[1])))
        gridz0=int(math.ceil(abs(self.origin[2]/self.spacing[2])))
        gridz1=int(math.ceil(abs(self.origin[2]*2/self.spacing[2])))
        
        step=np.array(self.spacing[:3])/spacingDivision
        start=self.coordMat[gridx0,gridy0,gridz0]+step*gap
        end=self.coordMat[self.shape[0]-gridx1,self.shape[1]-gridy1,self.shape[2]-gridz1]-step*gap+step/2.
        sampleCoord=[]
        for k in np.arange(start[0],end[0],step[0]):
            for l in np.arange(start[1],end[1],step[1]):
                for m in np.arange(start[2],end[2],step[2]):
                    sampleCoord.append(np.array([k,l,m]))
        return sampleCoord

        
    def getMatB(self,sampleCoord=None):
        '''
        get matrix B, basic bspline function, [m,n]
        '''
        print('getMatB......')
        flag=0
        if type(sampleCoord)==type(None):
            sampleCoord=self.sampleCoord
            flag=1
        n=np.prod(self.coefMat.shape[:3])
        dCList,CIndList=self.bsFourier.getdC(sampleCoord)    #len(self.BFpoints)=len(dCList)
        Basic=[]    #basic bspline value
        for i in range(len(dCList)):
            temp=sp.sparse.csr_matrix((dCList[i].reshape((-1,),order='F'),(np.zeros(len(CIndList[i])),CIndList[i].copy())),shape=(1,n))
            #tempRow=sp.sparse.hstack((temp,temp,temp),format='csr')
            #test
            tempRow=temp
            Basic.append(tempRow.copy())
        Basic=sp.sparse.vstack(Basic,format='csr')
        Basic=Basic.toarray()
        print('getMatB function done!!!')
        if flag==1:
            self.BMat=Basic
        else:
            return Basic
        
    def getMatdBdX(self,sampleCoord=None):
        '''
        get matrix dBdXMat, [m,n*uvw], related to sampleCoord and uvw
        '''
        print('getMatdC......')
        flag=0
        if type(sampleCoord)==type(None):
            sampleCoord=self.sampleCoord
            flag=1
        dCList,CIndList=self.bsFourier.getdC(sampleCoord,dxyz=[[1,0,0],[0,1,0],[0,0,1]])   # dCList [point,xyz,uvw], CIndList [point,xyz]

        print('getMatdBdX......')
        m=len(sampleCoord)
        n=np.prod(self.bsFourier.coef.shape[:3])
        uvw=self.bsFourier.coef.shape[4]
        dBdXMat=np.zeros((m,n*uvw))
        count=0
        ind=int(m/25)
        
        for num in range(m):
            if count%ind==0:
                print('getMatdBdX: ',count,' points calculated, processing ',count/m*100,'%')
            count+=1
            
            CInd=CIndList[num]
            dC=dCList[num]

            for j in range(uvw):
                for i in range(len(CInd)):
                    dBdXMat[num,j*n+CInd[i]]=dC[i,j]
        print('getMatdBdX function done!!!')
        if flag==1:
            self.dBdXMat=np.array(dBdXMat.copy())
        else:
            return np.array(dBdXMat)
        
    def getMatdUdX(self,coef=None,sampleCoord=None,order=None):
        '''
        matrix dUdXMat, solve divergence free equation
        get sum[(dB/dX)*C], related to sampleCoord, control point and uvw, fourier terms
        
        u=fu*(dg(t)/dt), fu=sum(B*C), g(t) is foureir basis function (cos or sin)
        error = du/dx + dv/dy + dw/dz
        if input time, the result is [FT, [du/dx, dv/dy, dw/dz]]
        if not input time, the result is [FT, [dfu/dx, dfv/dy, dfw/dz]]
        '''
        print('getMatdUdX......')
        #shape=self.bsFourier.coef.shape
        #basicBsplineFunc=[]
        if type(self.time)!=type(None):
            try:
                pass
            except:
                pass
        if type(coef)==type(None):
            resultdX=[]
            temp=[]
            count=0
            if type(sampleCoord)==type(None):
                sampleCoord=self.sampleCoord
            length=len(sampleCoord)
            ind=int(length/25)
            
            for coord in sampleCoord:
                dUdXMat=[]
                if count%ind==0:
                    print('getMatdUdX: ',count,' points calculated, processing ',count/length*100,'%')
                count+=1
                resultdX=self.bsFourier.getdX(coord)   # get dUdX, [dxdydz, FT, dudvdw] matrix, [3, FT, 3]
                for i in range(3):
                    dUdXMat.append(resultdX[i,:,i])
                dUdXMat=np.array(dUdXMat).transpose(1,0)  #get [FT, [du/dx,dv/dy,dw/dz]] from resultdX
                temp.append(dUdXMat.copy())   # [point, FT, uvw]
            print('getMatdUdX function done!!!')
            self.dUdXMat=np.array(temp.copy())   # [point, FT, uvw]
        else:
            uvw=3
            ft=self.coefMat.shape[3]
            n=np.prod(self.coefMat.shape[:3])    #number of control points
            if type(sampleCoord)==type(None):
                dBdXMat=self.dBdXMat
                sampleCoord=self.sampleCoord
            else:
                dBdXMat=self.getMatdBdX(sampleCoord)
            if order=='uvw':
                dUdX=np.zeros((len(sampleCoord),uvw))    #[sampleCoordSize,uvw], single fterm
                for i in range(uvw):
                    dUdX[:,i]=dBdXMat[:,i*n:(i+1)*n].dot(coef[i*n:(i+1)*n])
            elif order=='ft':
                dUdX=np.zeros((len(sampleCoord),ft))    #[sampleCoordSize,ft], single direction
                for i in range(ft):
                    dUdX[:,i]=dBdXMat[:,i*n:(i+1)*n].dot(coef[i*n:(i+1)*n])
            print('getMatdUdX function done!!!')
            return dUdX
        
    def errorCalc(self,sampleCoord=None,bgCoord=None,mode='displacementWise',fterm_start=1,weight=None,savePath=None,option=None):
        if type(sampleCoord)==type(None):
            sampleCoord=self.sampleCoord
        if type(bgCoord)==type(None):
            bgCoord=self.bgCoord
        if type(weight)==type(None):
            weight=self.weight
        if type(savePath)==type(None):
            savePath=self.savePath
        sampleCoordSize=len(sampleCoord)
        bgCoordSize=len(bgCoord)
        print('errorListCalc......')
        #print('sampleCoord has %d points, bgCoord has %d points, '%(sampleCoordSize,bgCoordSize),'weight is ', weight)
        #wdiag=np.ones(sampleCoordSize)
        n=np.prod(self.coefMat.shape[:3])    #number of control points
        ft=self.coefMat.shape[3]
        uvw=self.coefMat.shape[4]
        #for weight in self.weights:
            #wdiag=np.concatenate((wdiag,np.ones(3)*weight),axis=0)
        #if not(resume):
            #self.dCoef=[]
        #elif type(tempSave)!=type(None):
            #self.loadSamplingResults(tempSave)
        coefMat=self.coefMat.copy()      #[x,y,z,ft,uvw]
        dBdXMat=self.dBdXMat.copy()
        errorList=[]
        Basic=self.getMatB(bgCoord)    #len(bgCoord)=len(dCList)
        Buvw=np.zeros((3*bgCoordSize,3*n))
        for i in range(uvw):
            if type(weight[1]) in [np.ndarray,list]:
                Buvw[i*len(self.bgCoord):(i+1)*bgCoordSize,i*n:(i+1)*n]=Basic*weight[1][i]
            else:
                Buvw[i*len(self.bgCoord):(i+1)*bgCoordSize,i*n:(i+1)*n]=Basic*weight[1]
        Buvw=sp.sparse.csr_matrix(Buvw)
        #matW=sp.sparse.diags(np.ones(len(points)))
        for fterm in range(fterm_start,ft):
            if fterm>(ft//2):
                fw=fterm-ft//2
                print('Basic fourier function is sin(%dwt)------'%(fw))
            else:
                fw=fterm
                print('Basic fourier function is cos(%dwt)------'%(fw))
            if mode=='displacementWise':
                dogW=self.period/2      #coef for divFree equation
            elif mode=='velocityWise':
                dogW=((fw*2.*np.pi/self.period)**2)*self.period/2
            
            coef=np.array(coefMat[:,:,:,fterm,:]).reshape(-1,order='F')   #[n*uvw,1]
            #coef_start=coef.copy()
            dUdX=np.zeros((sampleCoordSize,uvw))    #[sampleCoordSize,uvw]
            dRdC=self.dBdXMat.copy()    #[sampleCoordSize,uvw*n]
            for i in range(uvw):
                dUdX[:,i]=dBdXMat[:,i*n:(i+1)*n].dot(coef[i*n:(i+1)*n])
            dUdX=np.sum(dUdX,axis=1)
            for ii in range(sampleCoordSize):
                dRdC[ii,:]*=(2*dogW*dUdX[ii]*weight[0])
            RMSMat=(dUdX**2)*dogW*weight[0]    #[sampleCoordSize,1]
            #rms=np.sum(RMSMat)
            ftCoefuvw=Buvw.dot(coef)    #[uvw*len(self.bgCoord),1]
            
            # key sentence
            RMSmeanSqrt=np.sqrt((RMSMat**2).mean())
            ftCoefmeanSqrt=np.sqrt((ftCoefuvw**2).mean())
            #valueList.append([RMSMat.max(),RMSMat.mean(),RMSMat.min(),ftCoefuvw.max(),ftCoefuvw.mean(),ftCoefuvw.min()])
            errorList.append([len(RMSMat),RMSMat.max(),RMSMat.mean(),RMSMat.min(),RMSmeanSqrt,len(ftCoefuvw),ftCoefuvw.max(),ftCoefuvw.mean(),ftCoefuvw.min(),ftCoefmeanSqrt])
        self.errorList=errorList
        if type(option)!=type(None):
            filePath=savePath+'errorList-{:s}.txt'.format(option)
        else:
            filePath=savePath+'errorList.txt'
        try:
            np.savetxt(filePath,self.errorList)
        except:
            pass
        return(errorList)
        
    def solve(self,method='ICP-disp',sampleCoord=None,bgCoord=None,regular='norm',maxError=0.0001,maxIteration=1000,fterm_start=1,weight=[1.,1.],convergence=0.8,reportevery=1000,saveFtermPath=None,tempSave=None,resume=False):
        '''
        method: 'ICP-disp', 'ICP-velc' (incompressible - displacementWise/velocityWise)
        regular: 'norm', 'fast', for ftCoefuvw (initial, update)
        Weight: Weight[0] for sampleCoord, Weight[1] for bgCoord
        '''
        self.weight=weight
        if method=='ICP-disp' or  method=='ICP-velc':
            mode=method[-4:]
            coefMat,rmsList=self.incompressibility(mode=mode,sampleCoord=sampleCoord,bgCoord=bgCoord,regular=regular,maxError=maxError,maxIteration=maxIteration,fterm_start=fterm_start,weight=weight,convergence=convergence,reportevery=reportevery,saveFtermPath=saveFtermPath,tempSave=tempSave,resume=resume)
        print('motionConstrain.solve completed')

    def incompressibility(self,mode=None,sampleCoord=None,bgCoord=None,regular='norm',maxError=0.0001,maxIteration=1000,weight=[1.,1.],fterm_start=1,convergence=0.5,reportevery=1000,saveFtermPath=None,tempSave=None,resume=False,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.):
        if sampleCoord==None:
            sampleCoord=self.sampleCoord
        if bgCoord==None:
            bgCoord=self.bgCoord
        sampleCoordSize=len(sampleCoord)
        bgCoordSize=len(bgCoord)
        print('method: ICP-{0:s}, regular: {1:s}'.format(mode,regular))
        print('sampleCoord has %d points, bgCoord has %d points, '%(sampleCoordSize,bgCoordSize),'weight is ', weight)
        #wdiag=np.ones(sampleCoordSize)
        n=np.prod(self.coefMat.shape[:3])    #number of control points
        ft=self.coefMat.shape[3]
        uvw=self.coefMat.shape[4]

        coefMat=self.coefMat.copy()      #[x,y,z,ft,uvw]
        dBdXMat=self.dBdXMat.copy()
        rmsList=[]
        Basic=self.getMatB(bgCoord)    #len(bgCoord)=len(dCList)
        Buvw=np.zeros((3*bgCoordSize,3*n))
        for i in range(uvw):
            if type(weight[1]) in [np.ndarray,list]:
                Buvw[i*len(self.bgCoord):(i+1)*bgCoordSize,i*n:(i+1)*n]=Basic*weight[1][i]
            else:
                Buvw[i*len(self.bgCoord):(i+1)*bgCoordSize,i*n:(i+1)*n]=Basic*weight[1]
        Buvw=sp.sparse.csr_matrix(Buvw)
        #matW=sp.sparse.diags(np.ones(len(points)))
        for fterm in range(fterm_start,ft):
            print('ICP_displacementWise......')
            if fterm>(ft//2):
                fw=fterm-ft//2
                print('Current basic fourier function is sin(%dwt)********************************'%(fw))
            else:
                fw=fterm
                print('Current basic fourier function is cos(%dwt)********************************'%(fw))
            
            if mode=='disp':
                dogW=self.period/2      #coef for divFree equation
            elif mode=='velc':
                dogW=((fw*2.*np.pi/self.period)**2)*self.period/2   #coef for divFree equation
            else:
                print('Error: please input correct method for incompressibility, ICP-disp or ICP-vlec')
                sys.exit()
            #dogW=self.period/2      #coef for divFree equation
            coef=np.array(coefMat[:,:,:,fterm,:]).reshape(-1,order='F')   #[n*uvw,1]
            #coef_start=coef.copy()
            dUdX=np.zeros((sampleCoordSize,uvw))    #[sampleCoordSize,uvw]
            dRdC=self.dBdXMat.copy()    #[sampleCoordSize,uvw*n]
            for i in range(uvw):
                dUdX[:,i]=dBdXMat[:,i*n:(i+1)*n].dot(coef[i*n:(i+1)*n])
            dUdX=np.sum(dUdX,axis=1)
            for ii in range(sampleCoordSize):
                dRdC[ii,:]*=(2*dogW*dUdX[ii]*weight[0])
            RMSMat=(dUdX**2)*dogW*weight[0]     #[sampleCoordSize,1]
            RMSMat = RMSMat/np.sqrt(self.period*self.spacingDivision[0]**3)    #normalized by the period and grid sampling  
            
            sparsedRdC=sp.sparse.csr_matrix(dRdC)
            A = sp.sparse.vstack([sparsedRdC,Buvw],format='csr')            
            
            ftCoefuvw_init=Buvw.dot(coef)    #[3*len(self.bgCoord),1]   
            ftCoefuvw_raw=ftCoefuvw_init
            ftCoefuvw=sp.sparse.csr_matrix(ftCoefuvw_raw).transpose()
            ftCoefuvw = ftCoefuvw/np.sqrt(self.spacingDivision[1]**3)            #normalized by the grid sampling
            Rfinal = RMSMat.copy()*0
            Rfinal=sp.sparse.csc_matrix(Rfinal).transpose()
            sparseRfianl=sp.sparse.vstack([Rfinal,ftCoefuvw],format='csc')
            
            sparseRMSMat = sp.sparse.csc_matrix(RMSMat).transpose()
            sparseRcalcu = sp.sparse.vstack([sparseRMSMat,ftCoefuvw],format='csc')
            residual = sparseRcalcu - sparseRfianl
            B = residual.copy()
            
            finalRMS = 0.5*residual.transpose().dot(residual).toarray()     # from sparse matrxi to np.array
            finalRMS = finalRMS[0,0]
            rmsList.append(6666666666666666)
            rmsList.append(finalRMS)            
            
            print('initial max(dRdC) is %.6f, initial max(coef) is %.4f, initial max(RMSMat) is %.6f'%(np.abs(dRdC).max(),np.abs(coef).max(),RMSMat.max()))
            error=float('inf')
            count=0.
            jumpFlag=0
    
            lmLambda=lmLambda_init
            reductionRatio=1.
            while error>maxError and count<maxIteration:
                print('iteration of solve_matrix of Fterm %d: %.4f-------------------------------'%(fterm,count))
                print('mean(coef) is {0:.8f}'.format(np.mean(coef)))
                
                matA = A.transpose().dot(A)
                temp=sp.sparse.diags(matA.diagonal())
                A_LM=sp.sparse.csc_matrix(matA+lmLambda*temp)     #lavenberg-Marquardt correction, add lmLambda*diagonals to the diagonal of matA
                matB = A.transpose().dot(B).toarray()
                
                A_LM=splu(A_LM)
                dCoef=A_LM.solve(matB) 
                dCoef = dCoef.reshape((-1,))  #[n*uvw]
                print('mean(dCoef) {0:f}'.format(np.mean(np.abs(dCoef))))
                #newCoef=newCoef.reshape((-1,))  #[n*uvw,1]
                #dCoef=newCoef-coef  #[n*uvw,1]
                
                coef_backup=coef.copy()
                A_backup=A.copy()
                B_backup=B.copy()
                #RMSMat_backup=RMSMat.copy()
                finalRMS_backup = finalRMS.copy()
                error_backup = error
                #ftCoefuvw_backup=ftCoefuvw_raw.copy()
                
                error=0
                minSpacing=self.spacing[:3].min()
                #coef[np.abs(coef)< minSpacing/1e6] = 0
                #newCoef[np.abs(newCoef)< minSpacing/1e6] = 0                                
                for i in range(dCoef.shape[0]):
                    if coef[i]==0:
                        error=max(error,np.abs(dCoef[i])/minSpacing)
                    else:
                        temp = np.abs(dCoef[i])/np.abs(coef[i])
                        #if round(error,8)==1.00000000:
                            #print('error {0:.2f}, dCoef[i] {1:.2f}, coef[i] {2:.2f}'.format(error, dCoef[i], coef[i]))
                        error=max(error,np.abs(dCoef[i])/np.abs(coef[i]))
                
                ratio=reductionRatio
                if convergence:
                    if abs(dCoef).max()>self.bsFourier.spacing[:3].min()*convergence:
                        ratio=min(ratio,self.bsFourier.spacing[:3].min()*convergence/abs(dCoef).max())                
                print('ratio: {0:.8f}; reductionRatio: {1:.8f}'.format(ratio,reductionRatio))
                                
                coef-=ratio*dCoef       # + or - decides by the form of residual
                dUdX=self.getMatdUdX(coef=coef,order='uvw') #[sampleCoordSize,uvw]
                dUdX=np.sum(dUdX,axis=1)
                dRdC=self.dBdXMat.copy()    #[sampleCoordSize,uvw*n]                                
                for ii in range(sampleCoordSize):
                    dRdC[ii,:]*=(2*dogW*dUdX[ii]*weight[0])
                RMSMat=(dUdX**2)*dogW*weight[0]    #[sampleCoordSize,1]
                RMSMat = RMSMat/np.sqrt(self.period*self.spacingDivision[0]**3)         #normalized by the period and grid sampling
                print('mean(dRdC) {0:f}'.format(np.mean(np.abs(dRdC)))) 
                print('mean(RMSMat_i) {0:f}'.format(np.mean(np.abs(RMSMat))))
                sparsedRdC=sp.sparse.csr_matrix(dRdC)
                A = sp.sparse.vstack([sparsedRdC,Buvw],format='csr')
                
                ftCoefuvw_raw=Buvw.dot(coef)    #[3*len(self.bgCoord),1]
                ftCoefuvw=sp.sparse.csr_matrix(ftCoefuvw_raw).transpose()   
                ftCoefuvw = ftCoefuvw/np.sqrt(self.spacingDivision[1]**3)                #normalized by the grid sampling
                
                if regular=='norm':
                    sparseRMSMat = sp.sparse.csc_matrix(RMSMat).transpose()
                    sparseRcalcu = sp.sparse.vstack([sparseRMSMat,ftCoefuvw],format='csc')
                    residual = sparseRcalcu - sparseRfianl
                    
                    finalRMS = 0.5*residual.transpose().dot(residual).toarray()
                    finalRMS = finalRMS[0,0]
                    B = residual.copy()
                    deltaRMS = finalRMS - finalRMS_backup
                
                elif regular=='greed':
                    sparseRMSMat_backup=sparseRMSMat
                    sparseRMSMat = sp.sparse.csc_matrix(RMSMat).transpose()
                    sparseRfianl_backup=sparseRfianl
                    sparseRfianl=sp.sparse.vstack([Rfinal,ftCoefuvw],format='csc')
                    
                    sparseRcalcu = sp.sparse.vstack([sparseRMSMat,ftCoefuvw],format='csc')
                    residual = sparseRcalcu - sparseRfianl
                    B = residual.copy()
                    deltaRMS=np.sum(sparseRMSMat-sparseRMSMat_backup)
                    error=np.abs(deltaRMS/np.sum(sparseRMSMat_backup))      #greedy method, only calculate the change of divergence as deltaRMS and error
                
                if deltaRMS>0.:
                    print('RMS value increases!!! finalRMS is %f\n      deltaRMS is %f, error is %.8f'%(finalRMS,deltaRMS,error))
                    coef=coef_backup.copy()
                    A=A_backup.copy()
                    B=B_backup.copy()
                    #sparsedRdC=sparsedRdC_backup.copy()
                    #RMSMat=RMSMat_backup.copy()
                    finalRMS = finalRMS_backup.copy()
                    error = error_backup
                    #ftCoefuvw_raw=ftCoefuvw_backup.copy()
                    if regular=='greed':
                        sparseRMSMat=sparseRMSMat_backup.copy()
                        sparseRfianl=sparseRfianl_backup.copy()
                        #ftCoefuvw=sp.sparse.csr_matrix(ftCoefuvw_raw).transpose()
                    #error=float('inf')
                    if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                        lmLambda*=lmLambda_incrRatio
                    else:
                        lmLambda=lmLambda_max
                        reductionRatio*=0.8
                    count+=0.02
                    
                    if jumpFlag==0:
                        jumpValue=np.sum(RMSMat)
                        jumpFlag=1
                        jumpNum=0
                    if jumpValue==np.sum(RMSMat):
                        jumpNum+=1
                    else:
                        jumpFlag==0
                    if jumpNum==20:
                        count=maxIteration  #jump out
                        print('RMS didnt change for %d iteration, Fterm %d jump out!!!!!!!!'%(jumpNum,fterm))
                else:
                    if regular=='norm':
                        print('RMS value is decreased!!! finalRMS is %f, deltaRMS is %f, error is %.8f'%(finalRMS,deltaRMS,error))
                    elif regular=='greed':
                        print('RMS value is decreased!!! finalRMS is %f, deltaRMS is %f, error is %.8f'%(np.sum(sparseRMSMat),deltaRMS,error))
                    if ratio>0.9 and lmLambda!=lmLambda_min:
                        if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                            lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                        else:
                            lmLambda=lmLambda_min
                    elif reductionRatio<0.9:
                        reductionRatio*=1.1
                    #rms=np.sum(RMSMat)  #new rms
                    count+=1
                
                if regular=='norm':
                    rmsList.append(finalRMS)
                elif regular=='greed':
                    rmsList.append(np.sum(sparseRMSMat))               
                #print('RMS is %.8f, error is %.8f'%(finalRMS,error))
                #print('max(dRdC) is %.6f, max(coef) is %.4f, max(RMSMat) is %.6f'%(np.abs(dRdC).max(),np.abs(coef).max(),RMSMat.max()))
                if count==maxIteration:
                    print('Maximum iterations reached for matrix: ft=',fterm)
                if (type(tempSave)!=type(None))and(int(count+1)%25==0):
                    self.writeSamplingResults(tempSave)
                    print('tempSave of matrix (ft=',fterm,') and iteration ',count)
            coefMat[:,:,:,fterm,:]=coef.reshape((*coefMat.shape[:3],uvw),order='F').copy()
            if type(saveFtermPath)!=type(None):
                np.savetxt((saveFtermPath+str(fterm)+'.txt'),coef,fmt='%.18f',delimiter=' ')
        rmsList=np.array(rmsList)
        self.coefMat=coefMat.copy()
        self.dCoefMat=self.coefMat-self.bsFourier.coef
        self.rmsList=rmsList.copy()
        return (coefMat,rmsList)
    
    def writeFile(self,filepath=None,coefMat=None,dCoefMat=None,divFreeMat=None,errorMat=None,RMSMat=None,dRdCMat=None,delimiter=' '):
        ''' 
        Write coef in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        FourierTerms=[]
        FourierTerms.append(self.bsFourier.coef.shape[3])
        if type(coefMat)!=type(None):
            np.savetxt(filepath,self.coefMat.reshape(-1, order='F'),delimiter=delimiter,comments='',header='(GridOrigin '+' '.join(map(str, self.bsFourier.origin))+')\n(GridSize '+' '.join(map(str, self.bsFourier.coef.shape))+')\n(GridSpacing '+' '.join(map(str, self.bsFourier.spacing))+')')
        
        if type(dCoefMat)!=type(None):
            # save order: FT, point
            np.savetxt(filepath,self.dCoefMat.reshape(-1, order='F'),delimiter=delimiter,comments='',header='(deltaCoef-GridOrigin '+' '.join(map(str, self.bsFourier.origin))+')\n(deltaCoef-GridSize '+' '.join(map(str, self.bsFourier.coef.shape))+')\n(deltaCoef-GridSpacing '+' '.join(map(str, self.bsFourier.spacing))+')')
            #np.savetxt(filepath,self.dRdCMat.reshape(-1, order='C'),delimiter=delimiter,comments='',header='(dRdCMatrixSize '+' '.join(map(str, self.dRdCMat.shape))+')\n(coefSize '+' '.join(map(str, self.bsFourier.coef.shape))+')')

        if type(divFreeMat)!=type(None):
            # save order: uvw, FT, point
            np.savetxt(filepath,self.divFreeMat.reshape(-1, order='F'),delimiter=delimiter,comments='',header='(sampleCoordSize '+' '.join(map(str, self.sampleCoordSize))+')\n(divFreeMatrixSize '+' '.join(map(str, self.divFreeMat.shape))+')\n(PointNumber '+' '.join(map(str, self.sampleCoord.shape))+')\n(FourierTerms '+' '.join(map(str, FourierTerms))+')')
        
        if type(errorMat)!=type(None):
            # save order: FT, point
            np.savetxt(filepath,self.errorMat.reshape(-1, order='F'),delimiter=delimiter,comments='',header='(errorMatrixSize '+' '.join(map(str, self.errorMat.shape))+')\n(PointNumber '+' '.join(map(str, self.sampleCoord.shape))+')\n(FourierTerms '+' '.join(map(str, FourierTerms))+')')
            # parameter insides join(map()) shouldn't be number
        
        if type(dRdCMat)!=type(None):
            # save order: FT, point
            np.savetxt(filepath,self.dRdCMat,fmt='%.8f',delimiter=delimiter,comments='',header='(dRdCMatrixSize '+' '.join(map(str, self.dRdCMat.shape))+')\n(coefSize '+' '.join(map(str, self.bsFourier.coef.shape))+')')
            #np.savetxt(filepath,self.dRdCMat.reshape(-1, order='C'),delimiter=delimiter,comments='',header='(dRdCMatrixSize '+' '.join(map(str, self.dRdCMat.shape))+')\n(coefSize '+' '.join(map(str, self.bsFourier.coef.shape))+')')
        
    def coefZeroRemap(self,coef=None,time=None):
        if type(coef)==type(None):
            coef=self.coefMat.copy()
        if type(time)!=type(None):
            coef[:,:,:,0]=0.
            for m in range(self.coefMat.shape[3]//2):#int(len(solver.pointsCoef[0])/2)):
                coef[:,:,:,0]-=coef[:,:,:,m+1]*np.cos((m+1.)*2.*np.pi/self.period*time)+coef[:,:,:,(self.coefMat.shape[3]//2)+m+1]*np.sin((m+1.)*2.*np.pi/self.period*time)
            self.coefMat=coef.copy()
            return coef
        else:
            print('coefZeroRemap Error: please input remap time point!!!')
        
    def coefCombiner(self,filePathList=None):
        newCoef=[]
        shape=list(self.coefMat.shape[:3])
        shape.append(self.coefMat.shape[-1])
        if self.coefMat.shape[3]-len(filePathList)==1:
            newCoef.append(self.coefMat[:,:,:,0,:])
        #filepath=[]
        #for fterm in range(1,(fourierTerms*2-1)):
            #filepath.append(saveFtermPath+str(fterm)+'.txt')
        for filePath in filePathList:#len(filepath)):
            temp=np.loadtxt(filePath,delimiter=' ').reshape(shape, order='F')
            newCoef.append(temp.copy())
        newCoef=np.array(newCoef).transpose(1,2,3,0,4)
        self.coefMat=newCoef.copy()
        return newCoef
        
    def writeSamplingResults(self,filepath,delimiter=' '):
        ''' 
        Write sampling results in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        saveMatrix=[]
        for n in range(len(self.points)):
            if len(self.pointsCoef)>n:
                saveMatrix.append([self.points[n][0],self.points[n][1],self.points[n][2],*self.pointsCoef[n].reshape(-1, order='F')])
            else:
                saveMatrix.append([self.points[n][0],self.points[n][1],self.points[n][2],*np.zeros(self.pointsCoef[0].size)])
        saveMatrix=np.array(saveMatrix)
        np.savetxt(filepath,saveMatrix,delimiter=delimiter,header=str(len(self.pointsCoef))+' points calculated-- Coordinates, Fourier u, Fourier v, Fourier w')

    def loadSamplingResults(self,filepath,delimiter=' '):   #to modify
        ''' 
        Read sampling results in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        self.points=[]
        self.pointsCoef=[]
        with open (filepath, "r") as myfile:
            data=myfile.readline()
        coeflen=[int(s) for s in data.split() if s.isdigit()][0]
        loadMatrix=np.loadtxt(filepath,delimiter=delimiter)
        for n in range(len(loadMatrix)):
            self.points.append(loadMatrix[n,:3].copy())
            if n<coeflen:
                self.pointsCoef.append(loadMatrix[n,3:].reshape((-1,3),order='F'))
