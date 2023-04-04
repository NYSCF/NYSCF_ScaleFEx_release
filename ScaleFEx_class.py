import numpy as np
import matplotlib.pyplot as plt
import os,glob
import pandas as pd
import scipy as sp 
from scipy import ndimage as ndi
import skimage
import cv2
import datetime
import pickle
import skimage.segmentation
import skimage.feature
import multiprocessing as mp
from datetime import datetime
import skimage.segmentation
import nuclei_location_extraction as nle
import utils
import warnings
import Compute_measurements_functions
warnings.filterwarnings('ignore')

class ScaleFEx:

    """ Pipeline to extract a vector of fixed features from a HCI screen

        Attributes: 
        
        exp_folder = string containing the experiment folder location (eg folder with subfolders of plates), str
        experiment_name = experiment name (eg ScaleFEx_xx), str
        saving_folder = string containing the destination folder, str 
        Plates = plates IDs to be analysed, as they appear in the pathname, list of strings 
        Channel = channel IDs to be analysed, as they appear in the pathname. NOTE: the nuclei stain has to be the firs channel of the list. list of strings 
        img_size = x and y size of the image, list of ints
        ROI = half the size of the cropped area around the cell, int
        parallel = use multiprocessing to analyse each plate with a worker, Bool
        SaveImage = Specify if to save the cropped images as a .npy file. False if not, pathname of saving location if yes
        stack = performs max projection on images if the acquisition mode was multi-stack, Bool
        CellType = choose between 'Fib' (Fibroblasts), 'IPSC' or 'Neuron'. A different segmentation algorithm is used based on this choice. str
        MitoCh = which of the channels is Mitochondria (if any), str
        RNAch = which of the channels is RNA (if any), str

    
    """
    
    
    def __init__(self,exp_folder,experiment_name='EXP',saving_folder='~/ScaleFEx_results/',Plates=['1','2','3','4','5'],
                 Channel=['ch4','ch1','ch2','ch3','ch5'],img_size=[2160,2160],ROI=299,parallel=True,SaveImage=False,stack=False, min_cell_size=False, max_cell_size=False, CellType='Fib',
                 MitoCh='ch2',RNAch='ch5',downsampling=1,visualization=False):
       
        self.stack=stack
        self.CellType=CellType
        self.exp_folder=exp_folder
        self.saving_folder=saving_folder
        self.Channel=Channel
        self.experiment_name=experiment_name
        self.SaveImage=SaveImage
        self.MitoCh=MitoCh
        self.RNAch=RNAch
        self.downsampling=downsampling
        self.viz=visualization
        self.ROI=int(ROI/downsampling)
        self.min_cell_size=min_cell_size
        self.max_cell_size=max_cell_size

        ### Reads the Flat Field corrected image if it exists, otherwise it computes it
        if not os.path.exists(self.saving_folder+experiment_name+'FFC.p'):
            print('Creating FFC image')
            files=pd.DataFrame(glob.glob(self.exp_folder+'/*/*.tiff'),columns=['filename'])
            self.FFC=utils.FFC_on_data(files,20,self.Channel)
            pickle.dump( self.FFC, open( self.saving_folder+experiment_name+'FFC.p', "wb" ) )

        else:
            print('Loading FFC image')
            self.FFC = pickle.load( open( self.saving_folder+experiment_name+'FFC.p', "rb" ) )
        
        self.img_size=[int(img_size[0]/downsampling),int(img_size[1]/downsampling)]
        if self.downsampling !=False:
            for key in self.FFC.keys():
                self.FFC[key]=cv2.resize(self.FFC[key],self.img_size)

        if parallel==True:
            processes = []
            for plate in Plates:
                
                p = mp.Process(target=self.cascade_functions, args=(plate,))
                #self.cascade_functions(plate)
                processes.append(p)
                p.start()
                # p.join()

            for process in processes:
                
                process.join()

        else:
            for plate in Plates:
                self.cascade_functions(plate)

               
    def cascade_functions(self,plate):
        
        files=utils.query_data(plate,self.exp_folder)
        Wells,fields=utils.make_well_and_field_list(files)
        csv_file=self.saving_folder+'FeatureVector/'+self.experiment_name+'_'+str(plate)+'FeatureVector.csv'
        if not os.path.exists(self.saving_folder+'FeatureVector'):
            os.makedirs(self.saving_folder+'FeatureVector')
        flag,ind,Wells,Site_ex,flag2=utils.check_if_file_exists(csv_file,Wells,fields[-1])
        self.vector_extraction_Phoen(files,plate,Wells,Site_ex,flag2,fields,csv_file,flag,ind)  

    def vector_extraction_Phoen(self,files,plate,Wells,Site_ex,flag2,fields,csv_file,flag,ind):
       
        if flag!='over':
            for well in Wells:
                print(well,plate,datetime.now())
                for site in fields:
                    
                   
                    if site==Site_ex or flag2==False:
                        flag2=False
                        print(site,well, plate,datetime.now())
                        Vector=pd.DataFrame()
                    
                                            
                        #images=np.zeros((self.img_size[0],self.img_size[1],len(self.Channel)))
                        np_images = []
                        if self.stack==True:
                            np_images=utils.process_Zstack(image_fnames,self.Channel,np_images)
                        else:
                            corrupt=False
                            for ch in self.Channel:
                                
                                image_fnames = files.loc[(files.Well==well) & (files.Site==site) & (files.channel==ch),'file_path'].values[0] 
                                if self.stack!=True:
                                    img= utils.load_image(image_fnames) 
                                else:
                                    img=utils.process_Zstack(image_fnames)
                                if ch==self.Channel[0]:
                                    imgNuc=img.copy()
                                if self.downsampling != 1:
                                    img=cv2.resize(img,self.img_size)
                                    
                                if (img is not None) and (img.shape[0]==self.img_size[0]) and (img.shape[1]==self.img_size[1]): #Check that the image is of the right format
                                    #if ch!='ch6':
                                    img=img/self.FFC[ch]
                                    
                                    img=(img/(np.max(img))) * 255
                                    np_images.append(img.astype('uint8'))
                                   
                                else:
                                    corrupt=True
                                    print('Img corrupted')

                        if corrupt==False:   
                            np_images = np.array(np_images)
                            np_images = np.expand_dims(np_images, axis=3)
                            scale=1
                            if self.CellType=='Neuron':
                                CoM2=nle.Count_DAPI_Neurons_Coords(imgNuc)
                            if self.CellType=='iPSC':
                                CoM2=nle.Count_DAPI_IPSCs(imgNuc)
                            else:
                                
                                CoM2=nle.retrieve_coordinates(nle.compute_DNA_mask(imgNuc),CellSizeMin=self.min_cell_size*self.downsampling,CellSizeMax=self.max_cell_size/self.downsampling)  

                            try:
                                CoM2
                            except NameError:
                                CoM2 = [] 
                                print('No Cells detected')
                                
                            if len(CoM2)>2:
                                
                                for cn,CoM in enumerate(CoM2):
                                    CoM=[CoM[0]*(scale/self.downsampling),CoM[1]*(scale/self.downsampling)]
                                    if (int(CoM[0]-self.ROI)>0) and (int(CoM[0]+self.ROI)<self.img_size[0]) and (int(CoM[1]-self.ROI)>0) and (int(CoM[1]+self.ROI)<self.img_size[1]):
                                        ccDistance=[]
                                        for cord in CoM2:  
                                            ccDistance.append(sp.spatial.distance.pdist([CoM, cord]))   
                                            ccDistance.sort()  
                                        Crop=np.zeros((self.ROI*2,self.ROI*2,len(np_images)))
                                        if self.SaveImage:
                                            np.save(self.SaveImage+plate+well+site+str(cn)+'.npy',np_images)
                                        for iii in range(len(np_images)):     
                                            Crop[:,:,iii]=np_images[iii][int(CoM[0]-self.ROI):int(CoM[0]+self.ROI),int(CoM[1]-self.ROI):int(CoM[1]+self.ROI),0]
                                    
                                        print(CoM)
                                        fla,Vector=self.single_cell_feature_extraction(Crop,self.Channel)
                                        Vector.index=[ind]
                                        if fla==True:
                                          

                                            Vector.loc[ind,'Well']=well
                                            Vector.loc[ind,'Site']=site
                                            Vector.loc[ind,'Cell_ID']=cn+1
                                            Vector.loc[ind,'Cell_Num']=len(CoM2)
                                            Vector.loc[ind,'Close_Cell_1']=ccDistance[1]
                                            Vector.loc[ind,'Close_Cell_2']=ccDistance[2]
                                            Vector.loc[ind,'CoordX']=CoM[0]
                                            Vector.loc[ind,'CoordY']=CoM[1]
                     
                                            Vector.to_csv(csv_file[:-4]+'.csv',mode='a',header=flag)

                                            flag=False
                                            ind+=1

                                                                   

                            else:
                                flag=False
                        else:
                                flag=False
        else:
            print('plate ',plate, 'is done')

    def single_cell_feature_extraction(self,simg, channels):

        ROI=self.ROI
        Lab={}
        regions={}
        Cat=pd.DataFrame([[]])
        
        for i in range(len(channels)):
            chan=channels[i]

            Lab[i]=Compute_measurements_functions.compute_primary_mask(simg[:,:,i])
            invMask=Lab[i]<1
            if i==0:
                nn=Lab[i][self.ROI,self.ROI]
                Lab[i]=Lab[i]==nn

            else:
                nn=Lab[i]*ndi.binary_dilation(Lab[0], structure=skimage.morphology.disk(int(100/self.downsampling)))
               
                try:
                    nn=np.bincount(nn[nn>0]).argmax()
                except:
                    print('out except')
                    return False,False
          
                Lab[i] = Lab[i] == nn
            if self.viz==True:
                utils.show_cells([simg[:,:,i],Lab[i]],title=[chan +'_'+str(i),'mask'])

            if np.count_nonzero(Lab[i]) <= 50/self.downsampling:
                print('out size')
                return False,False 

            a=simg[:,:,i]*Lab[i]
            b=a[a>0]

            a=simg[:,:,i]*invMask

            c=a[a>0]

            SNR = np.mean(b)/np.std(c)
            Cat['SNR_intensity%s' % chan] = SNR
            regions[i] = skimage.measure.regionprops(Lab[i].astype(int))
         
 
            ### Shape

            Cat=pd.concat([Cat,Compute_measurements_functions.compute_shape(chan,regions[i],ROI,Lab[i])],axis=1)
 
            ### Texture
            Cat=pd.concat([Cat,Compute_measurements_functions.iter_Text(chan,simg[:,:,i],Lab[i],ndistance=5,nangles=4)],axis=1)
            Cat=pd.concat([Cat,Compute_measurements_functions.Texture_single_values(chan,Lab[i],simg[:,:,i])],axis=1)
            
        
            # Granularity

            Cat=pd.concat([Cat,Compute_measurements_functions.Granularity(chan,simg[:,:,i],n_convolutions=16)],axis=1)
            
            # Intensity
            Cat=pd.concat([Cat,Compute_measurements_functions.Intensity(simg[:,:,i],Lab[i],chan,regions[i])],axis=1)

            ## Concentric measurements

            scale=8
            if chan == channels[0]:
                nuc=Cat['MaxRadius_shape'+channels[0]].values[0] * 0.1
            else:
                nuc=0
            Cat=pd.concat([Cat,Compute_measurements_functions.concentric_measurements(scale,ROI,simg[:,:,i],Lab[i],chan,DAPI=nuc)],axis=1)

        ## Mitochondria measurements

            if chan == self.MitoCh:
                Cat=pd.concat([Cat,Compute_measurements_functions.Mitochondria_measurement(Lab[i],simg[:,:,i],viz=self.viz)],axis=1)
           
        ## RNA measurements

            if chan == self.RNAch:
                Cat=pd.concat([Cat,Compute_measurements_functions.RNA_measurement(Lab[0],simg[:,:,i],viz=self.viz)],axis=1)

        # Colocalization

        for i in range(len(channels)):
            chan=channels[i]
            for j in range(i + 1, len(channels) - 1):
                Cat=pd.concat([Cat,Compute_measurements_functions.Correlation_measurements(simg[:,:,i], simg[:,:,j],chan,channels[j],Lab[i],Lab[j])],axis=1)

        fla=True
        return fla,Cat

