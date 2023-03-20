import glob,os,cv2,random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def query_data(plate,exp_folder):
    files= glob.glob(exp_folder+'/*'+plate+'*/Images/*.tiff')
    files.sort()
    files=pd.DataFrame(files,columns=['file_path'])
    files['filename']=[i[i.find('/r')+1:] for i in files.file_path]
    files['Well']=[i[i.find('r'):i.find('r')+6] for i in files.filename]
    files['Site']=[i[i.find('f'):i.find('f')+3] for i in files.filename]
    files['channel']=[i[i.find('ch'):i.find('ch')+3] for i in files.filename]
    return files



def make_well_and_field_list(files):
    
    Wells=np.unique(files.Well)
    Wells.sort()
    fields=np.unique(files.Site)
    fields.sort()
    return Wells,fields

def check_if_file_exists(csv_file,Wells,last_field):

    Site_ex=1
    flag2=False
    if os.path.exists(csv_file):
        Vector=pd.read_csv(csv_file,index_col=0,header=0)
    
        indSS=Vector.index[-1]+1
        
        lastWell=Vector.Well[Vector.index[-1]]
        Site_ex=Vector.Site[Vector.index[-1]]

        if (lastWell==Wells[-1]) and (Site_ex==last_field):
            return 'over', indSS,Wells,Site_ex,True

        Wells=Wells[np.where(Wells==lastWell)[0][0]:]

        if Site_ex!=last_field:
            Site_ex='f'+str(int(Site_ex[1:])+1).zfill(2)
        else:
            Site_ex='f01'
            Wells=Wells[np.where(Wells==lastWell)[0][0]+1:]

        flag=False
        flag2=True
        
    else:
        flag=True
        
        indSS=0
        

    
    return flag,indSS,Wells,Site_ex,flag2
    


def load_image(file_path):

    im=cv2.imread(file_path,-1)
    return im



def FFC_on_data(files,n_images,Channel):
    FFC={}
    for ch in Channel:
        
        B=files.sample(n_images)
        img=cv2.imread(B.iloc[0].filename,-1)
        for i in range(1,n_images):
            img=np.stack([cv2.imread(B.iloc[i].filename,-1),img],axis=2)
            img=np.min(img,axis=2)
        FFC[ch]=img
    return FFC

def process_Zstack(image_fnames,Channel,np_images):
    for ch in Channel:
        image_fnames = files[0] ### TO BE FIXED ### sorted(glob.glob(+well+'f'+str(site).zfill(2)+'*'+ch+'*.tiff'))
        imgCh=[]
        for name in image_fnames:
            imgCh.append(load_image(name))/FFC[ch]
        imgCh=np.max(np.asarray(imgCh),axis=0)
        imgCh=(imgCh/(np.max(imgCh)))*255
        np_images.append(imgCh.astype('uint8'))
    return np_images

def show_cells(images,title=[''],size=4):
    _,ax=plt.subplots(1,len[images],figsize=(int(size*len(images)),size))
    for i in range(len(images)):
        ax[i].imshow(images[0])
        if len(title)==len(images):
            ax[i].set_title(title[i])
        else:
            ax[0].set_title(title[0])
        ax[i].axis('off')