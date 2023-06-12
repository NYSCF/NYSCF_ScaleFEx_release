''' Functions for data handling'''
import glob
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def query_data(plate, exp_folder):
    ''' Queries the data from the folders and extracts wells, sites and channels. 
        This is the main function to be changed if the user's has the files 
        arranged in a different way. The output is a dataframe that contains well, 
        site, channel, file name and file path of each image '''

    files = glob.glob(exp_folder+'*'+plate+'*/*/*.tiff')
    files.sort()
    files = pd.DataFrame(files, columns=['file_path'])
    files['filename'] = [i[i.find('/r')+1:] for i in files.file_path]
    files['Well'] = [i[i.find('r'):i.find('r')+6] for i in files.filename]
    files['Site'] = [i[i.find('f'):i.find('f')+3] for i in files.filename]
    files['channel'] = [i[i.find('ch'):i.find('ch')+3] for i in files.filename]
    files['plane'] = [i[i.find('p'):i.find('p')+3] for i in files.filename]
    return files


def make_well_and_field_list(files):
    ''' inspects the image file name and extracts the unique fields and wells to loop over'''
    wells = np.unique(files.Well)
    wells.sort()
    fields = np.unique(files.Site)
    fields.sort()
    return wells, fields


def check_if_file_exists(csv_file, wells, last_field):
    ''' Checks if a file for the plate and experiment exists. if it does, if checks what is 
        the last well and field that was calculated. If it equals the last available well and field,
        it considers the computation over, otherwise it extracts where is stops and takes over 
        from there '''
    site_ex = 1
    flag2 = False
    if os.path.exists(csv_file):
        fixed_feature_vector = pd.read_csv(csv_file, index_col=0, header=0)

        ind = fixed_feature_vector.index[-1]+1

        last_well = fixed_feature_vector.Well[fixed_feature_vector.index[-1]]
        site_ex = fixed_feature_vector.Site[fixed_feature_vector.index[-1]]

        if (last_well == wells[-1]) and (site_ex == last_field):
            return 'The computation is over', ind, wells, site_ex, True

        wells = wells[np.where(wells == last_well)[0][0]:]

        if site_ex != last_field:
            site_ex = 'f'+str(int(site_ex[1:])+1).zfill(2)
        else:
            site_ex = 'f01'
            wells = wells[np.where(wells == last_well)[0][0]+1:]

        flag = False
        flag2 = True

    else:
        flag = True

        ind = 0

    return flag, ind, wells, site_ex, flag2


def load_image(file_path):
    ''' image loader'''
    im = cv2.imread(file_path, -1)
    return im


def flat_field_correction_on_data(files, n_images, channel):
    ''' Calculates the background trend of the entire experiment to be used for flat field correction'''
    flat_field_correction = {}
    for ch in channel:

        B = files.sample(n_images)
        img = load_image(B.iloc[0].filename)
        for i in range(1, n_images):
            img = np.stack([load_image(B.iloc[i].filename), img], axis=2)
            img = np.min(img, axis=2)
        flat_field_correction[ch] = img
    return flat_field_correction


def process_zstack(image_fnames):
    ''' Computes the stack's max projection from the image neame'''
    img = []
    for name in image_fnames:
        img.append(load_image(name))
    img = np.max(np.asarray(img), axis=0)

    return img


def show_cells(images, title=[''], size=3):
    ''' Function to visualize  images in a compact way '''
    _, ax = plt.subplots(1, len(images), figsize=(int(size*len(images)), size))
    if len(images) > 1:
        for i,_ in enumerate(images):
            ax[i].imshow(images[i], cmap='Greys_r')
            if len(title) == len(images):
                ax[i].set_title(title[i])
            else:
                ax[0].set_title(title[0])
            ax[i].axis('off')
    else:
        ax.imshow(images[0])
        ax.set_title(title[0])
        ax.axis('off')
    plt.show()
