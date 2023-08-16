'''Main class for the ScaleFEx computations'''
import datetime
import pickle
import cv2
import utils
import os
import glob
import warnings
import numpy as np
import pandas as pd
import scipy as sp
import skimage.segmentation
import skimage.feature
import multiprocessing as mp
from datetime import datetime
import nuclei_location_extraction as nle

import compute_measurements_functions
warnings.filterwarnings('ignore')


class ScaleFEx:

    """ Pipeline to extract a vector of fixed features from a HCI screen

        Attributes: 

        exp_folder = string containing the experiment folder location (eg folder with subfolders of plates), str
        experiment_name = experiment name (eg ScaleFEx_xx), str
        saving_folder = string containing the destination folder, str 
        plates = plates IDs to be analysed, as they appear in the pathname, list of strings 
        channel = channel IDs to be analysed, as they appear in the pathname. 
            NOTE: the nuclei stain has to be the firs channel of the list. list of strings 
        img_size = x and y size of the image, list of ints
        roi = half the size of the cropped area around the cell, int
        parallel = use multiprocessing to analyse each plate with a worker, Bool
        save_image = Specify if to save the cropped images as a .npy file. False if not,
            pathname of saving location if yes
        stack = performs max projection on images if the acquisition mode was multi-stack, Bool
        CellType = choose between 'Fib' (Fibroblasts), 'IPSC' or 'Neuron'. 
            A different segmentation algorithm is used based on this choice. str
        mito_ch = which of the channels is mito_chondria (if any), str
        rna_ch = which of the channels is RNA (if any), str


    """

    def __init__(self, exp_folder, experiment_name='EXP', saving_folder='~/output/',
                 plates=['1', '2', '3', '4', '5'], channel=['ch4', 'ch1', 'ch2', 'ch3', 'ch5'],
                 img_size=[2160, 2160], save_image=False, stack=False,
                 min_cell_size=False, max_cell_size=False, mito_ch='ch2', rna_ch='ch5',neuritis_ch='',
                 downsampling=1, visualization=False, roi=150,location_csv=False,max_processes=60):

        self.stack = stack
        self.exp_folder = exp_folder
        self.saving_folder = saving_folder
        self.channel = channel
        self.experiment_name = experiment_name
        self.save_image = save_image
        self.mito_ch = mito_ch
        self.rna_ch = rna_ch
        self.neuritis_ch = neuritis_ch
        self.downsampling = downsampling
        self.viz = visualization
        self.roi = int(roi/downsampling)
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.location_csv=location_csv

        # Reads the Flat Field corrected image if it exists, otherwise it computes it
        if not os.path.exists(self.saving_folder+experiment_name+'FFC.p'):
            print('Creating flat_field_correction image')
            files = pd.DataFrame(
                glob.glob(self.exp_folder+'/**/*.tiff',recursive=True), columns=['filename'])
            self.flat_field_correction = utils.flat_field_correction_on_data(
                files, 200, self.channel)
            pickle.dump(self.flat_field_correction, open(self.saving_folder +
                        experiment_name+'FFC.p', "wb"))

        else:
            print('Loading flat_field_correction image')
            self.flat_field_correction = pickle.load(
                open(self.saving_folder+experiment_name+'FFC.p', "rb"))

        self.img_size = [int(img_size[0]/downsampling),
                         int(img_size[1]/downsampling)]
        if self.downsampling is not False:
            for key in self.flat_field_correction.keys():
                self.flat_field_correction[key] = cv2.resize(
                    self.flat_field_correction[key], self.img_size)


        for plate in plates:
            self.cascade_functions(plate,max_processes)

    def cascade_functions(self, plate,max_processes):
        ''' Function that calls all the functions to compute single cell fixed features 
            on all the images within a plate'''
        files = utils.query_data(plate, self.exp_folder)
        wells, fields = utils.make_well_and_field_list(files)
        csv_file = self.saving_folder+'FeatureVector/' + \
            self.experiment_name+'_'+str(plate)+'FeatureVector.csv'
        if not os.path.exists(self.saving_folder+'FeatureVector'):
            os.makedirs(self.saving_folder+'FeatureVector')
        if os.path.exists(csv_file):
            self.retrieve=True
            wells = utils.check_if_file_exists(
                csv_file, wells, fields[-1])
        else:
            self.retrieve=False

        #Parallelesation

        # Create a multiprocessing Manager Queue to hold the tasks
        task_queue = mp.Manager().Queue()   

        # Function to add tasks to the queue
        def add_task_to_queue(task):
            task_queue.put(task)

        # Iterate over wells to generate tasks
        for well in wells:
            task = well # Create the task using well information
            add_task_to_queue(task)

        def process_worker(semaphore):
            while True:
                semaphore.acquire()  # Acquire a permit from the semaphore
                if task_queue.empty():
                    semaphore.release()
                    break  # If the queue is empty, break the loop
                task = task_queue.get()
                self.load_preprocess_and_compute_feature(files, plate, task, 
                                            fields, csv_file)
                semaphore.release()  # Release the permit
    
        # Create a Semaphore with the maximum number of allowed processes
        process_semaphore = mp.Semaphore(max_processes)

        # Start the worker processes
        processes = []
        for _ in range(max_processes):
            p = mp.Process(target=process_worker, args=(process_semaphore,))
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()
            p.close()  # Close the process (release resources)

    

    def load_preprocess_and_compute_feature(self, files, plate, well,
                                            fields, csv_file):
        ''' Function that imports the images and extracts the location of cells'''
        if well[0] == 'Over':
            print('plate ', plate, 'is done')
            return
        ind=0
        if self.retrieve is True:
                well_sites = pd.read_csv(csv_file,usecols=['Well','Site','Cell_ID']).sort_values(by=['Well','Site','Cell_ID'])
                computed_sites = well_sites.loc[well_sites.Well==well]
                fields=np.delete(fields,np.where(np.isin(fields,computed_sites)))
                ind=len(well_sites)
        
        #for well in wells:
        print(well, plate, datetime.now())
        for site in fields:   
   
            print(site, well, plate, datetime.now())
            single_cell_vector = pd.DataFrame()

            np_images = []
            corrupt = False
            for ch in self.channel:

                image_fnames = files.loc[(files.Well == well) & (
                    files.Site == site) & (files.channel == ch), 'file_path'].values
                if self.stack is not True:
                    img = utils.load_image(image_fnames[0])
                else:
                    img = utils.process_zstack(image_fnames)

                if ch == self.channel[0]:
                    imgNuc = img.copy()
                if self.downsampling != 1:
                    img = cv2.resize(img, self.img_size)

                # Check that the image is of the right format
                if (img is not None) and (img.shape[0] == self.img_size[0]) and (img.shape[1] == self.img_size[1]):
                    img = img/self.flat_field_correction[ch]

                    img = (img/(np.max(img))) * 255
                    np_images.append(img.astype('uint8'))

                else:
                    corrupt = True
                    print('Img corrupted')

            if corrupt is False:
                np_images = np.array(np_images)
                np_images = np.expand_dims(np_images, axis=3)
                scale = 1

                # extraction of the location of the cells
                if self.location_csv is False:
                    center_of_mass = nle.retrieve_coordinates(nle.compute_DNA_mask(imgNuc),
                                                                cell_size_min=self.min_cell_size*self.downsampling,
                                                                cell_size_max=self.max_cell_size/self.downsampling)

                    try:
                        center_of_mass
                    except NameError:
                        center_of_mass = []
                        print('No Cells detected')
                else:
                    locations=pd.read_csv(self.location_csv,index_col=0)
                    locations['Plate']=locations['Plate'].astype(str)
                    locations=locations.loc[(locations.Well==well)&(locations.Site==site)&(locations.Plate==plate)]
                    center_of_mass=np.asarray(locations[['CoordX','CoordY']])
                if len(center_of_mass) > 0:
                    field_vec=pd.DataFrame()
                    for cn, com in enumerate(center_of_mass):
                        com = [
                            com[0]*(scale/self.downsampling), com[1]*(scale/self.downsampling)]
                        if ((int(com[0]-self.roi) > 0) and (int(com[0]+self.roi) < self.img_size[0])
                                and (int(com[1]-self.roi) > 0) and (int(com[1]+self.roi) < self.img_size[1])):
                            cell2cell_distance = []
                            for cord in center_of_mass:
                                cell2cell_distance.append(
                                    sp.spatial.distance.pdist([com, cord]))
                                cell2cell_distance.sort()
                            cell_crop = np.zeros(
                                (self.roi*2, self.roi*2, len(np_images)))
                            if self.save_image:
                                np.save(
                                    self.save_image+plate+well+site+str(cn)+'.npy', np_images)
                            
                            for iii,_ in enumerate(np_images):
                                cell_crop[:, :, iii] = np_images[iii][int(
                                    com[0]-self.roi):int(com[0]+self.roi), int(com[1]-self.roi):int(com[1]+self.roi), 0]

                            quality_flag, single_cell_vector = compute_measurements_functions.single_cell_feature_extraction(
                                cell_crop, self.channel,self.roi,self.mito_ch,self.rna_ch,self.neuritis_ch,self.downsampling,self.viz)

                            if quality_flag is True:

                                single_cell_vector.index = [ind]
                                single_cell_vector.loc[ind,
                                                        'Well'] = well
                                single_cell_vector.loc[ind,
                                                        'Site'] = site
                                if self.location_csv is False:
                                    single_cell_vector.loc[ind,
                                                            'Cell_ID'] = cn+1
                                    single_cell_vector.loc[ind, 'Cell_Num'] = len(
                                        center_of_mass)
                                    single_cell_vector.loc[ind,
                                                            'distance'] = cell2cell_distance[1]
                                    # single_cell_vector.loc[ind,
                                    #                         'Close_Cell_2'] = cell2cell_distance[2]
                                else:
                                    single_cell_vector.loc[ind,
                                                            'distance'] = locations['distance'].values[cn]  # cell2cell_distance[1]
                                    # single_cell_vector.loc[ind,
                                    #                         'Close_Cell_2'] = cell2cell_distance[2]
                                single_cell_vector.loc[ind,
                                                            'Cell_ID'] = locations['Cell_ID'].values[cn]
                                single_cell_vector.loc[ind,
                                                        'CoordX'] = com[0]
                                single_cell_vector.loc[ind,
                                                        'CoordY'] = com[1]
                                field_vec=pd.concat([field_vec,single_cell_vector],axis=0)
                    if os.path.exists(csv_file[:-4]+'.csv'):
                        flag=False
                    else:
                        flag=True
                    ind+=1
                    field_vec.to_csv(
                        csv_file[:-4]+'.csv', mode='a', header=flag)

                   

    
