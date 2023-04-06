
from scipy import ndimage as ndi
import numpy as np
import skimage

def compute_DNA_mask(DNAimg):
    '''Given the input grayscale DNA image DNAimg returns the mask of the segmented objects

        Attributes:

        DNAimg: grayscale image of the DNA channel, numpy array

        Returns:

        Lab: labelled image of the segmented cells
        '''
    
    Am = DNAimg > skimage.filters.threshold_li(DNAimg)

    Am = ndi.binary_opening(Am, structure=skimage.morphology.disk(3),iterations=1)
    Am = ndi.binary_fill_holes(Am)

    Lab = skimage.measure.label(Am)


    return Lab

def retrieve_coordinates(Lab,CellSizeMin=1000, CellSizeMax=50000):
    '''Given the labelled image Lab returns the coordinates of the centroids (CoM) of the nuclei
        
        Attributes:

        Lab: image containing labelled segmented objects. Array 
        CellSizeMin: Minimum size for the object to be considered a nuclei. int
        CellSizeMax: Maximum size for the object to be considered a nuclei. int
        
        Returns:

        CoM: coordinates of the Center of Mass of the segmented objects, list. 
             Length is the number of objets, width is 2 (X and Y coordinates)
         '''              
    labGF=skimage.measure.label(Lab)
    for i in range(1,labGF.max()+1):
        if np.count_nonzero(labGF==i)<CellSizeMin or np.count_nonzero(labGF==i)>CellSizeMax:
            labGF[labGF==i]=0
    labels=skimage.measure.label(labGF)
    CoM=np.zeros((labGF.max(),2))
    for id in np.arange(np.max(labels)):
        CoM[id]=[ndi.measurements.center_of_mass(labels==id+1)[0],ndi.measurements.center_of_mass(labels==id+1)[1]]

    return CoM
