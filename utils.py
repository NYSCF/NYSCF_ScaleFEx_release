''' Functions for data handling'''
import glob,os,cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import time
import requests
from PIL import Image
from io import BytesIO,StringIO
import io

def query_data(plate, exp_folder):
    ''' Queries the data from the folders and extracts wells, sites and channels. 
        This is the main function to be changed if the user's has the files 
        arranged in a different way. The output is a dataframe that contains well, 
        site, channel, file name and file path of each image '''

    files = glob.glob(exp_folder+'/*_'+plate+'_*/**/*.tiff')
    files.sort()
    print(files[0])
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
    if os.path.exists(csv_file):
        fixed_feature_vector = pd.read_csv(csv_file, usecols=['Well','Site','Cell_ID'])
        fixed_feature_vector=fixed_feature_vector.sort_values(by=['Well','Site','Cell_ID'])
        last_well = fixed_feature_vector.Well.values[-1]
        #site_ex = fixed_feature_vector.Site.values[-1]

        if (last_well == wells[-1]) and (fixed_feature_vector.loc[fixed_feature_vector.Well==last_well,'Site'][-1] == last_field):
            return ['Over']

        wells = wells[np.where(wells == last_well)[0][0]:]


    return  wells


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

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------S3 Functions------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

def check_if_files_exist_s3(bucket_name, csv_key, Wells, last_field):

    s3 = boto3.client('s3')
    # Check if the file exists in the S3 bucket
    try:
        s3.head_object(Bucket=bucket_name, Key=csv_key)
        file_exists = True
    except Exception as e:
        file_exists = False
    Site_ex = 1
    flag2 = False
    if file_exists:
        # Download the CSV file from S3 to a StringIO object
        csv_data = StringIO()
        s3.download_fileobj(bucket_name, csv_key, csv_data)
        csv_data.seek(0)
        Vector = pd.read_csv(csv_data, index_col=0, header=0)
        indSS = Vector.index[-1] + 1
        lastWell = Vector.Well[Vector.index[-1]]
        Site_ex = Vector.Site[Vector.index[-1]]
        if (lastWell == Wells[-1]) and (Site_ex == last_field):
            return 'over', indSS, Wells, Site_ex, True
        Wells = Wells[np.where(Wells == lastWell)[0][0]:]
        if Site_ex != last_field:
            Site_ex = 'f' + str(int(Site_ex[1:]) + 1).zfill(2)
        else:
            Site_ex = 'f01'
            Wells = Wells[np.where(Wells == lastWell)[0][0] + 1:]
        flag = False
        flag2 = True
    else:
        flag = True
        indSS = 0
  
    return flag, indSS, Wells, Site_ex, flag2

def FFC_on_data_S3(bucket_name, n_images, Channel, experiment_name):
    
    FFC = {}
    files = process_data_files_for_ffc(bucket_name, experiment_name)
    for ch in Channel:
        B = files.sample(n_images)
        
        img = read_image_from_s3(bucket_name, B.iloc[0].filename)
        for i in range(1, n_images):

            img = np.stack([read_image_from_s3(bucket_name, B.iloc[i].filename), img], axis=2)
            img = np.min(img, axis=2)
        FFC[ch] = img
 
    return FFC

def read_image_from_s3(bucket, object_name):

    s3 = boto3.client('s3')
    image_data = BytesIO()
    #Download and store the image as a binary object
    s3.download_fileobj(bucket, object_name, image_data)
    image_data.seek(0)
    im = cv2.imdecode(np.asarray(bytearray(image_data.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return im

def query_data_files_df(bucket_name, plate, experiment_name):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket_name):
        for content in page.get('Contents', []):
            if content['Key'].endswith('.tiff') and experiment_name in content['Key'] and  '_' + plate + '_' in content['Key'] :
                files.append(content['Key'])
    files.sort()
    files_df = pd.DataFrame(files, columns=['file_path'])
    files_df['filename'] = [i[i.find('/r')+1:] for i in files_df.file_path]
    files_df['Well'] = [i[i.find('r'):i.find('r')+6] for i in files_df.filename]
    files_df['Site'] = [i[i.find('f'):i.find('f')+3] for i in files_df.filename]
    files_df['channel'] = [i[i.find('ch'):i.find('ch')+3] for i in files_df.filename]
  
    return files_df

def query_data_files_df_for_FFC(bucket_name, experiment_name):

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket_name):
        for content in page.get('Contents', []):
            if content['Key'].endswith('.tiff') and experiment_name in content['Key'] :
                files.append(content['Key'])
    files.sort()
    files_df = pd.DataFrame(files, columns=['file_path'])
    files_df['filename'] = [i[i.find('/r')+1:] for i in files_df.file_path]
    files_df['Well'] = [i[i.find('r'):i.find('r')+6] for i in files_df.filename]
    files_df['Site'] = [i[i.find('f'):i.find('f')+3] for i in files_df.filename]
    files_df['channel'] = [i[i.find('ch'):i.find('ch')+3] for i in files_df.filename]
    return files_df

def process_data_files_for_ffc(bucket_name, experiment_name):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    tiff_files = []
    for page in paginator.paginate(Bucket=bucket_name):
        for content in page.get('Contents', []):
            if content['Key'].endswith('.tiff') and experiment_name in content['Key'] :
                tiff_files.append(content['Key'])
    files = pd.DataFrame(tiff_files, columns=['filename'])
    return files


def process_data_files(bucket_name, experiment_name, plate, saving_folder, Channel):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    tiff_files = []
    for page in paginator.paginate(Bucket=bucket_name):
        for content in page.get('Contents', []):
            if content['Key'].endswith('.tiff') and experiment_name in content['Key'] and  '_' + plate + '_' in content['Key'] :
                tiff_files.append(content['Key'])
    files = pd.DataFrame(tiff_files, columns=['filename'])
    return files

def csv_to_s3(csv_file):
    s3 = boto3.client('s3')
    s3.upload_file(csv_file, 'nyscf-feature-vector', 'results/'+ str(csv_file).replace("/","_"))

def process_zstack_s3(bucket_name, image_keys):
    '''Computes the stack's max projection from the image names in an S3 bucket'''
    s3 = boto3.client('s3')
    img = []
   
    for key in image_keys:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response['Body'].read()
        image = Image.open(io.BytesIO(image_bytes))
        img.append(np.array(image))
    img = np.max(np.asarray(img), axis=0)
    return img

def query_data_files_results(bucket_name, plate, experiment_name):
    ''' Checks for all the already computed wells by looking at the results csv 
    existing in the S3 bucket
    Returns the list of already existing well's features
    '''
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=bucket_name):
        for content in page.get('Contents', []):
            if content['Key'].endswith('.csv') and experiment_name in content['Key'] and  '_' + plate + '_' in content['Key'] and 'results' in content['Key']:
                files.append(content['Key'])
    files.sort()
    files_df = pd.DataFrame(files, columns=['file_path'])
    files_df['Well'] = [i[i.find('r'):i.find('r')+6] for i in files_df.filename]
  
    return files_df


def check_if_files_exists_s3(bucket_name, csv_key, Wells, last_field):

    s3 = boto3.client('s3')
    # Check if the file exists in the S3 bucket
    try:
        s3.head_object(Bucket=bucket_name, Key=csv_key)
        file_exists = True
    except Exception as e:
        file_exists = False
    Site_ex = 1
    flag2 = False
    if file_exists:
        # Download the CSV file from S3 to a StringIO object
        csv_data = StringIO()
        s3.download_fileobj(bucket_name, csv_key, csv_data)
        csv_data.seek(0)
        Vector = pd.read_csv(csv_data, index_col=0, header=0)
        indSS = Vector.index[-1] + 1
        lastWell = Vector.Well[Vector.index[-1]]
        Site_ex = Vector.Site[Vector.index[-1]]
        if (lastWell == Wells[-1]) and (Site_ex == last_field):
            return 'over', indSS, Wells, Site_ex, True
        Wells = Wells[np.where(Wells == lastWell)[0][0]:]
        if Site_ex != last_field:
            Site_ex = 'f' + str(int(Site_ex[1:]) + 1).zfill(2)
        else:
            Site_ex = 'f01'
            Wells = Wells[np.where(Wells == lastWell)[0][0] + 1:]
        flag = False
        flag2 = True
    else:
        flag = True
        indSS = 0
  
    return flag, indSS, Wells, Site_ex, flag2

def check_if_file_exists(csv_file, wells, last_field):
    ''' Checks if a file for the plate and experiment exists. if it does, if checks what is 
        the last well and field that was calculated. If it equals the last available well and field,
        it considers the computation over, otherwise it extracts where is stops and takes over 
        from there '''
    if os.path.exists(csv_file):
        fixed_feature_vector = pd.read_csv(csv_file, usecols=['Well','Site','Cell_ID'])
        fixed_feature_vector=fixed_feature_vector.sort_values(by=['Well','Site','Cell_ID'])
        last_well = fixed_feature_vector.Well.values[-1]
        #site_ex = fixed_feature_vector.Site.values[-1]

        if (last_well == wells[-1]) and (fixed_feature_vector.loc[fixed_feature_vector.Well==last_well,'Site'][-1] == last_field):
            return ['Over']

        wells = wells[np.where(wells == last_well)[0][0]:]

    return  wells

def get_instance_id():
    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    return response.text

def terminate_current_instance():
    ec2 = boto3.client('ec2')
    instance_id = get_instance_id()

    ec2.terminate_instances(InstanceIds=[instance_id])















# def list_missing_cells(csv_output,location,plate):
    
#     location_csv = location.loc[location['Plate'] == int(plate)]
#     merged = location_csv.merge(csv_output, on=['Plate','Well','Site','Cell_ID'], how='left', indicator=True)
#     missing_cells = merged[merged['_merge'] == 'left_only']
#     missing_cells = missing_cells.drop(columns='_merge')
#     location_csv = missing_cells[location_csv.columns]
#     csv_to_s3(location_csv)
#     return location_csv

# def check_csv_output(bucket_name,location, experiment_name, plate):
#     """
#     Check if a file exists in an S3 bucket.
#     :param bucket_name: Name of the S3 bucket
#     :param file_key: Key (path) of the file in the S3 bucket
#     """
#     s3 = boto3.client('s3')
    
#     paginator = s3.get_paginator('list_objects_v2')
#     # location_csv = pd.read_csv(location,index_col=0)
    
#     print(bucket_name,experiment_name, plate)
#     try:
#         s3.head_object(Bucket=bucket_name, Key=experiment_name+'_' + plate+'_'+'FeatureVector.csv')
#         # s3.download_file(bucket_name, experiment_name+'_' + plate+'_'+'FeatureVector.csv', experiment_name+'_' + plate+'_'+'FeatureVector.csv')
#         # print('Output_csv found')
#         # csv_output = pd.read_csv(experiment_name+'_' + plate+'_'+'FeatureVector.csv')
#         # location_csv = list_missing_cells(csv_output,location_csv,plate)
#         # print('Output_csv found, location_csv updated, processing of the remaining cells will start')

#     except Exception as e:
#         csv_output = experiment_name+'_'+plate+'_'+'FeatureVector.csv'
#         f = open(csv_output, "x")
#         print('No output_csv found, processing of the whole dataset will start')
#     # return csv_output,location_csv
#     return csv_output