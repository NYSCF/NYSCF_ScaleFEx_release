import ScaleFEx_class
import numpy as np
path_to_images='~/data/' # Data path of the images.The code will look for a folder per experiment
                          # The structure would be /folder_path/experiment_name*Platen*/r01c01f01-ch1*.tif
saving_folder='/~/HCI_Projects/' #folder where the csv containing the measurements is saved
                                #It will create a folder called 'FeatureVector' in the saving folder
experiment_name='ScaleFEX' #experiment name, it will look for it in the filepath
plates=['Plate1','Plate2','Plate3','Plate4','Plate5'] #Plate names, it will look for it in the filepath
ROI=150  #radius to define the crop-size 
channels=['ch4','ch1','ch2','ch3','ch5'] #channels to be analysed and in that order. 
                                         #DNA should be first as it defines the center of the crop 
MitoCh='ch2'  # Mitochondria channel. If you don't want to compute the Mito fearues, put ''
RNAch='ch5'   # RNA channel. If you don't want to compute the RNA fearues, put ''
image_size=[2160,2160] # X-Y image size
downsampling=1  # downsampling ratio
parallel=True   # parallelization flag. If True, it will parallelaize the computation over 6 workers
visualization=False # Visualizatin flag: change it to True if you want to see all the computed masks
SaveImage=False # Put True if you want to save the crops as .npy in a folder
stack=False  # Indicate here if the images were acquired in stacks
max_cell_size=500000 # max area in pixels for nuclei segmentation
min_cell_size=1000 #min area in pixels for nuclei segmentation. These values are set for a 40X image of size 2160x2160

ScaleFEx_class.ScaleFEx(path_to_images,saving_folder=saving_folder,experiment_name=experiment_name,
                        Plates=plates,Channel=channels,ROI=ROI,visualization=visualization,
                        img_size=image_size,parallel=parallel,SaveImage=SaveImage,stack=stack, 
                        MitoCh=MitoCh,RNAch=RNAch,downsampling=downsampling,max_cell_size=max_cell_size,min_cell_size=min_cell_size)


