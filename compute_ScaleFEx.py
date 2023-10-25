''' File used to launch the sctipt. First imput the configurations for your experiment, 
    then launch this file'''
import argparse
import ScaleFEx_cloud_class
import ScaleFEx_on_prem_class


# Data path of the images.The code will look for a folder per experiment
PATH_TO_IMAGES = '/media/biancamigliori/3b0129e9-916e-4029-a23f-615712223e70/HTF0003/'
# The structure would be /folder_path/experiment_name*Plate_n*/r01c01f01-ch1*.tif

ROI = 150  # radius to define the crop-size
# channels to be analysed and in that order.
channels = ['ch1','ch2','ch3','ch4','ch5']
# DNA should be first as it defines the center of the crop
# Mitochondria channel. If you don't want to compute the Mito fearues, put ''
MITO_CHANNEL = 'ch2'
RNA_CHANNEL = 'ch5'   # RNA channel. If you don't want to compute the RNA fearues, put ''
NEURON_CHANNEL='' # Neuron channel. Computes neuritis length, nodes and If you don't want to compute the RNA fearues, put ''
image_size = [2160, 2160]  # X-Y image size
DOWNSAMPLING = 1  # downsampling ratio

# Visualizatin flag: change it to True if you want to see all the computed masks
VISUALIZATION = False
SAVE_IMAGE = False  # Put True if you want to save the crops as .npy in a folder
STACK = False  # Indicate here if the images were acquired in stacks
MAX_CELL_SIZE = 50000  # max area in pixels for nuclei segmentation
# min area in pixels for nuclei segmentation. These values are set for a 40X image of size 2160x2160
MIN_CELL_SIZE = 1000

MAX_PROCESSES=10 # number of processes to use to parallelize the computation

def ScaleFEx_Main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', # the experiment name is used for path finding
						help="experiment's name to be processed ex: 'ScaleFEX'",
                        default="HTF0003")
    parser.add_argument('-p', '--plate', # put here all the plates you want to analyze in square brackets
						help="plate ID if only one plate is to be processed ex: ['Plate1']",
                        default=['201','202','203','204','205','206'])
    parser.add_argument('-r', '--ressource', # specify weather the computation happends locally or on AWS servers
						help="whether you want to use your machine or AWS. options: 'cloud' or 'on premise",
						default="on premise")
    
    
    args = vars(parser.parse_args())
    EXPERIMENT_NAME = args['experiment']
    print(EXPERIMENT_NAME)
    PLATE = args['plate']
    print(PLATE)
    RESSOURCE = args['ressource']

    if RESSOURCE == 'cloud':
        print(RESSOURCE)
        SAVING_FOLDER =  '/home/ec2-user/project/'
        CSV_COORDINATES='' # insert here the path to a CSV containing the coordinates of the cells to be analyzed. If empty, the segmentation method will locate the cells
        ScaleFEx_cloud_class.ScaleFEx(RESSOURCE,PATH_TO_IMAGES, saving_folder=SAVING_FOLDER,experiment_name=EXPERIMENT_NAME,
                        plate=PLATE, channel=channels, roi=ROI, visualization=VISUALIZATION,
                        img_size=image_size, save_image=SAVE_IMAGE, stack=STACK,
                        mito_ch=MITO_CHANNEL, rna_ch=RNA_CHANNEL, downsampling=DOWNSAMPLING,
                        max_cell_size=MAX_CELL_SIZE, min_cell_size=MIN_CELL_SIZE, location_csv=CSV_COORDINATES,bucket='nyscf-feature-vector')
        
    else:
        print(RESSOURCE)
        SAVING_FOLDER = '/home/biancamigliori/Documents/HCI_Projects/INAD/Vector/'
        CSV_COORDINATES='/home/biancamigliori/Documents/HCI_Projects/Runs/HTF0003/HTF0003_coordinates.csv' # insert here the path to a CSV containing the coordinates of the cells to be analyzed. If empty, the segmentation method will locate the cells
        ScaleFEx_on_prem_class.ScaleFEx(PATH_TO_IMAGES, saving_folder=SAVING_FOLDER,experiment_name=EXPERIMENT_NAME,
                        plates=PLATE, channel=channels, roi=ROI, visualization=VISUALIZATION,
                        img_size=image_size,  save_image=SAVE_IMAGE, stack=STACK,max_processes=MAX_PROCESSES,
                        mito_ch=MITO_CHANNEL, rna_ch=RNA_CHANNEL,neuritis_ch=NEURON_CHANNEL, downsampling=DOWNSAMPLING,
                        max_cell_size=MAX_CELL_SIZE, min_cell_size=MIN_CELL_SIZE, location_csv=CSV_COORDINATES)

	
if __name__ == "__main__":
	ScaleFEx_Main()
