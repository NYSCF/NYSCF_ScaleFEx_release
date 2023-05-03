''' File used to launch the sctipt. First imput the configurations for your experiment, 
    then launch this file'''
import ScaleFEx_class
# Data path of the images.The code will look for a folder per experiment
PATH_TO_IMAGES = '~/data/'
# The structure would be /folder_path/experiment_name*Plate_n*/r01c01f01-ch1*.tif
# folder where the csv containing the measurements is saved
SAVING_FOLDER = '/~/FeatureVector/'
# It will create a folder called 'FeatureVector' in the saving folder
EXPERIMENT_NAME = 'ScaleFEX'  # experiment name, it will look for it in the filepath
# Plate names, it will look for it in the filepath
plates = ['Plate1', 'Plate2', 'Plate3', 'Plate4', 'Plate5']
ROI = 150  # radius to define the crop-size
# channels to be analysed and in that order.
channels = ['ch4', 'ch1', 'ch2', 'ch3', 'ch5']
# DNA should be first as it defines the center of the crop
# Mitochondria channel. If you don't want to compute the Mito fearues, put ''
MITO_CHANNEL = 'ch2'
RNA_CHANNEL = 'ch5'   # RNA channel. If you don't want to compute the RNA fearues, put ''
image_size = [2160, 2160]  # X-Y image size
DOWNSAMPLING = 1  # downsampling ratio
# parallelization flag. If True, it will parallelaize the computation over 6 workers
PARALLEL = True
# Visualizatin flag: change it to True if you want to see all the computed masks
VISUALIZATION = False
SAVE_IMAGE = False  # Put True if you want to save the crops as .npy in a folder
STACK = False  # Indicate here if the images were acquired in stacks
MAX_CELL_SIZE = 500000  # max area in pixels for nuclei segmentation
# min area in pixels for nuclei segmentation. These values are set for a 40X image of size 2160x2160
MIN_CELL_SIZE = 1000

ScaleFEx_class.ScaleFEx(PATH_TO_IMAGES, saving_folder=SAVING_FOLDER,experiment_name=EXPERIMENT_NAME,
                        plates=plates, Channel=channels, ROI=ROI, visualization=VISUALIZATION,
                        img_size=image_size, parallel=PARALLEL, SaveImage=SAVE_IMAGE, stack=STACK,
                        MitoCh=MITO_CHANNEL, RNAch=RNA_CHANNEL, downsampling=DOWNSAMPLING,
                        max_cell_size=MAX_CELL_SIZE, min_cell_size=MIN_CELL_SIZE)
