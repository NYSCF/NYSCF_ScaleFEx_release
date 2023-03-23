
from scipy import ndimage as ndi
import numpy as np
import skimage

def compute_DNA_mask(DNAimg):

    Am = ndi.gaussian_filter((DNAimg), 3)
    filter_Am = ndi.gaussian_filter(Am, 1)
    alpha = 10
    Am = Am + alpha * (Am - filter_Am)
    Am = Am > skimage.filters.threshold_multiotsu(Am)[0]*0.9

    Am = ndi.binary_closing(Am, structure=skimage.morphology.disk(2))
    Am = ndi.binary_fill_holes(Am)

    Lab = skimage.measure.label(Am)

    return Lab

def Count_DAPI_IPSCs(img3,CellSize=20,fact=1):
    imgDen = ndi.gaussian_filter(img3, 3)
    Am2 = imgDen > skimage.filters.threshold_multiotsu(imgDen)[0] * fact
    #Am2 = ndi.binary_fill_holes(Am2)
    Am2=skimage.segmentation.clear_border(Am2)
    Am2=ndi.binary_opening(Am2,structure=skimage.morphology.disk(2),iterations=2)

    
    #what is the difference between the "fact" and "binary opening" dont they both enlarge the nuclei
    
    labGF=skimage.measure.label(Am2)
    count=0
    for i in range(1,labGF.max()+1):
        if np.count_nonzero(labGF==i)<CellSize:
            labGF[labGF==i]=0
            
        elif np.count_nonzero(labGF==i)>1000:
            count+=int(np.count_nonzero(labGF==i)/800)
        else:
            count+=1
    
    '''fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax=ax.ravel()
    ax[0].imshow(Am2)
    ax[1].imshow(img3)
    ax[1].set_title(count)
    plt.show()'''
    
    return count

def Count_DAPI_Opt(img3,CellSize=5000,CellSizeMax=30000,fact=1):
    imgDen = ndi.gaussian_filter(img3, 3)
    if len(np.unique(imgDen))>1 and np.all(np.isnan(imgDen))==False:
        Am2 = imgDen > skimage.filters.threshold_otsu(imgDen) * fact
        Am2 = ndi.binary_fill_holes(Am2)
        Am2=skimage.segmentation.clear_border(Am2)
        Am2=ndi.binary_opening(Am2,structure=skimage.morphology.disk(2),iterations=4)
        #plt.imshow(Am2)
        
        #what is the difference between the "fact" and "binary opening" dont they both enlarge the nuclei
        
        labGF=skimage.measure.label(Am2)
        for i in range(1,labGF.max()+1):
            if np.count_nonzero(labGF==i)<CellSize or np.count_nonzero(labGF==i)>CellSizeMax:
                labGF[labGF==i]=0
        labGF=skimage.measure.label(labGF)
    
        return np.max(labGF)
    else:
        return 0
def Count_DAPI_Extract_Coords(img3,CellSize=500,CellSizeMax=30000,fact=1):

    imgDen = ndi.gaussian_filter(img3, 3)
    if len(np.unique(imgDen))>1:
        Am2 = imgDen > skimage.filters.threshold_otsu(imgDen) * fact
        Am2 = ndi.binary_fill_holes(Am2)
        Am2=skimage.segmentation.clear_border(Am2)
        Am2=ndi.binary_opening(Am2,structure=skimage.morphology.disk(2),iterations=4)
        #plt.imshow(Am2)
                    
        labGF=skimage.measure.label(Am2)
        for i in range(1,labGF.max()+1):
            if np.count_nonzero(labGF==i)<CellSize or np.count_nonzero(labGF==i)>CellSizeMax:
                labGF[labGF==i]=0
        labels=skimage.measure.label(labGF)
        CoM=np.zeros((labGF.max(),2))
        for id in np.arange(np.max(labels)):
            CoM[id]=[ndi.measurements.center_of_mass(labels==id+1)[0],ndi.measurements.center_of_mass(labels==id+1)[1]]
    else:
        CoM=[]
    return CoM