import numpy as np
import skimage,cv2
from scipy import ndimage as ndi
import pandas as pd
import scipy as sp 
import utils

def compute_primary_mask(simg):

    Am = ndi.gaussian_filter((simg), 3)
    filter_Am = ndi.gaussian_filter(Am, 1)
    alpha = 10
    Am = Am + alpha * (Am - filter_Am)
    Am = Am > skimage.filters.threshold_multiotsu(Am)[0]*0.9

    Am = ndi.binary_closing(Am, structure=skimage.morphology.disk(2))
    Am = ndi.binary_fill_holes(Am)

    Lab = skimage.measure.label(Am)

    return Lab


def compute_shape(chan,regions,ROI,Lab):
    df=pd.DataFrame([[]])
    df['MinRadius_shape%s' % chan] = np.min([regions[0].bbox[2]-regions[0].bbox[0],regions[0].bbox[3]-regions[0].bbox[1]])
    df['MaxRadius_shape%s' % chan] = np.max([regions[0].bbox[2]-regions[0].bbox[0],regions[0].bbox[3]-regions[0].bbox[1]])
    df['MeanRadius_shape%s' % chan] = regions[0].equivalent_diameter
    df['Area_shape%s' % chan] = np.nansum(Lab)
    df['Perimeter_shape%s' % chan] = regions[0].perimeter
    df['FormFactor_shape%s' % chan] = (4 * np.pi * np.nansum(Lab)) / (regions[0].perimeter) ** 2
    df['Solidity_shape%s' % chan] = np.nansum(Lab) / regions[0].convex_area
    df['Extent_shape%s' % chan] = np.nansum(Lab) / ((2 * ROI) * (2 * ROI))
    df['Eccentricity_shape%s' % chan] = regions[0].eccentricity
    df['Orientation_shape%s' % chan] = regions[0].orientation
    df['Compactness_shape%s' % chan] = df['MeanRadius_shape%s' % chan]/ np.nansum(Lab)
    return df

def iter_Text(chan,simg,Lab,ndistance=5,nangles=4):
    df=pd.DataFrame([[]])
    angles = np.linspace(0,np.pi,num=nangles)
    distances=np.linspace(5,5*ndistance,num=ndistance).astype(int)
    for dcount,dis in enumerate(distances):
        for angle in angles:
            
            TextPropsA = skimage.feature.texture.graycomatrix(np.uint8(simg * Lab) * 255, [dis], [angle])           
            df['Texture_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(skimage.feature.texture.graycoprops(TextPropsA, prop='ASM'))
            df['TextContrast_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(
                skimage.feature.texture.graycoprops(TextPropsA, prop='contrast'))
            df['TextCorrelation_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(
                skimage.feature.texture.graycoprops(TextPropsA, prop='correlation'))
            df['TextDissimilarity_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(
                skimage.feature.texture.graycoprops(TextPropsA, prop='dissimilarity'))
            df['TextHomo_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(
                skimage.feature.texture.graycoprops(TextPropsA, prop='homogeneity'))
            df['TextEnergy_dist_' + str(dcount) + 'angle' + str(round(angle,2)) + chan] = np.nanmean(
                skimage.feature.texture.graycoprops(TextPropsA, prop='energy'))
       
    return df

def Texture_single_values(chan,Lab,simg):
    df=pd.DataFrame([[]])
    df['Variance_Texture' + chan] = np.nanvar((Lab * simg) > 0)
    df['Variance_Sum_Ave_Texture' + chan] = np.nansum((Lab * simg) > 0) / (
                ((Lab * simg) > 0).shape[0] * ((Lab * simg) > 0).shape[1])
    df['Variance_Ave_Texture' + chan] = np.nanvar((Lab * simg) > 0) / (
                ((Lab * simg) > 0).shape[0] * ((Lab* simg) > 0).shape[1])
    temp=((Lab * simg) - 1)/((Lab * simg) - 1).max()
    temp[temp<-1]=-1
    A4 = skimage.filters.rank.entropy(temp, skimage.morphology.disk(3))
    df['Variance_Entropy_Texture' + chan] = np.nanmean(A4 > 0)
    return df

def Granularity(chan,simg,n_convolutions=16):
    df=pd.DataFrame([[]])
    for gr in range(1, n_convolutions):
        df['Granularity_' + chan + '_' + str(gr)] = np.nanmean(ndi.morphology.grey_opening(simg, size=(gr, gr)))
    return df

def Intensity(simg,Lab,chan,regions):
            
    df=pd.DataFrame([[]])

    nanA = simg * Lab
    
    nanA[nanA == 0] = np.nan

    df['Integrated_intensity_' + chan] = np.nansum(nanA)  # The sum of the pixel intensities within an object.

    df['Mean_intensity_' + chan] = np.nanmean(nanA)  #: The average pixel intensity within an object.

    df['Std_intensity_' + chan] = np.nanstd(nanA)  #: The standard deviation of the pixel intensities within an object.

    df['Max_intensity_' + chan] = np.nanmax(nanA)  #: The maximal pixel intensity within an object.

    df['Min_intensity_' + chan] = np.nanmin(nanA)  # The minimal pixel intensity within an object.

    df['UpperQuartile_intensity_' + chan] = np.nanquantile(nanA, 0.75)  # : The intensity value of the pixel for which 75% of the pixels in the object have lower values.

    df['LowerQuartile_intensity_' + chan] = np.nanquantile(nanA, 0.25)  #: The intensity value of the pixel for which 25% of the pixels in the object have lower values.


    df['Median_intensity_' + chan] = np.nanmedian(nanA)  #: The median intensity value within the object.

    Edge = (ndi.sobel(Lab) != 0) * simg
    Edge[Edge == 0] = np.nan

    df['Integrated_intensityEdge_' + chan] = np.nansum(Edge)  #: The sum of the edge pixel intensities of an object.

    df['Mean_intensityEdge_' + chan] = np.nanmean(Edge)  #: The average edge pixel intensity of an object.

    df['Std_intensityEdge_' + chan] = np.nanstd(
        Edge)  #: The standard deviation of the edge pixel intensities of an object.

    df['Max_intensityEdge_' + chan] = np.nanmax(Edge)  #: The maximal edge pixel intensity of an object.

    df['Min_intensityEdge_' + chan] = np.nanmin(Edge)  #: The minimal edge pixel intensity of an object.

    df['MassDisplacement_intensity' + chan] = sp.spatial.distance.pdist([ndi.measurements.center_of_mass(simg, Lab), regions[0].centroid])[0]  #: The distance between the centers of gravity in the gray-level representation of the object and the binary representation of the object.

    
    df['MAD_intensity_' + chan] = np.nanmedian(np.abs(nanA-df['Median_intensity_' + chan].values))  #: The median absolute deviation (MAD) value of the intensities within the object. The MAD is defined as the median(|xi - median(x)|).

    
    df['Location_CenterMass_intensity_X' + chan] = ndi.measurements.center_of_mass(simg, Lab)[0]  # , Location_CenterMassIntensity_Y: The (X,Y) coordinates of the intensity weighted centroid (= center of mass = first moment) of all pixels within the object.

    df['Location_CenterMass_intensity_Y' + chan] = ndi.measurements.center_of_mass(simg, Lab)[1]  #: The (X,Y) coordinates of the pixel with the maximum intensity within the object.
    
    return df

def create_concentric_areas(scale,fact,ROI,DAPI=0):
       
    P = np.zeros((int(ROI)*2, int(ROI)*2))
    P2 = np.zeros((int(ROI)*2, int(ROI)*2))
    Dma = fact*(ROI/scale) 
    cv2.circle(P, (int(np.ceil(ROI)), int(np.ceil(ROI))), int(Dma), 1, -1)
    if fact!=0:
        cv2.circle(P2, (int(np.ceil(ROI)), int(np.ceil(ROI))), int(Dma + Dma), 1, -1)
    else:
        cv2.circle(P2, (int(np.ceil(ROI)), int(np.ceil(ROI))), int((ROI/scale)), 1, -1)

    return(P2 - P)
        

def concentric_measurements(scale,ROI,simg,Lab,chan,DAPI=0):
    imgConc = {}
    Pt = {}
    df=pd.DataFrame([[]])
    for fact in range(0, scale):
        if DAPI!=0:
            P=create_concentric_areas(scale,fact,ROI,DAPI=DAPI)
        else:
            P = create_concentric_areas(scale,fact,ROI)

        imgConc[fact] = (P * (simg * Lab))
        Pt[fact] = np.nansum(P * Lab)

        if np.nansum(Pt[fact]) > 0:
            df['Concent_tot_intensity_' + str(fact) + chan] = np.nansum(imgConc[fact]) / np.nansum(P)
            df['Concent_mean_intensity_' + str(fact) + chan] = np.nanmean(imgConc[fact]) / np.nansum(P)
            df['Concent_variation_intensity_' + str(fact) + chan] = np.nanstd(imgConc[fact]) / np.nansum(P)
            for gr in range(1, 16):
                df['Granularity_' + chan + '_' + str(gr) + '_Conc' + str(fact)] = np.nanmean(
                    ndi.morphology.grey_opening(imgConc[fact] / Pt[fact], size=(gr, gr)))
        else:
            df['Concent_tot_intensity_' + str(fact) + chan] = 0
            df['Concent_mean_intensity_' + str(fact) + chan] = 0
            df['Concent_variation_intensity_' + str(fact) + chan] = 0

            for gr in range(1, 16):
                df['Granularity_' + chan + '_' + str(gr) + '_Conc' + str(fact)] = 0

    for fact in range(1, 7):

        for fact2 in range(fact + 1, 8):

            if np.nansum(Pt[fact]) > 0 and np.nansum(Pt[fact2]) > 0:
                RR = np.concatenate([np.asarray(imgConc[fact]).reshape(-1, ) / Pt[fact],
                                    np.asarray(imgConc[fact2]).reshape(-1, ) / Pt[fact2]])
                df['Concent_Radial_intensity_' + str(fact) + '_' + str(fact2) + '_' + chan] = np.nanmean(
                    sp.stats.variation(RR))
            else:
                df['Concent_Radial_intensity_' + str(fact) + '_' + str(fact2) + '_' + chan] = 0

    return df

def Mitochondria_measurement(Lab,simg,viz=False):
    df=pd.DataFrame([[]])
    mito = ndi.gaussian_filter((simg), 1)
    filter_mito = ndi.gaussian_filter(mito, 1)
    alpha = 30
    mito = mito + alpha * (mito - filter_mito)
    
    Mseg = mito > skimage.filters.threshold_multiotsu(mito)[-1]
    Mseg=Mseg*Lab
    skel = skimage.morphology.skeletonize(Mseg)
    LabSkel = skimage.morphology.label(skel)
    for u in range(1, np.max(LabSkel)):
        if np.nansum(LabSkel == u) < 5:
            LabSkel[LabSkel == u] = 0
    LabSkel = skimage.morphology.label(LabSkel)
    if np.count_nonzero(LabSkel) < 1:
        df['MitoCount'] = 0
        df['MitoVolumeMean'] = 0
        df['MitoVolumeTot'] = 0
        df['MitoVolumeSkel'] = 0
        branch = []
        AspectRatio = []
        EndPoints = []
    else:
        df['MitoCount'] = np.max(LabSkel)
        df['MitoVolumeMean'] = np.count_nonzero(Mseg) / np.max(LabSkel)
        df['MitoVolumeTot'] = np.count_nonzero(Mseg)
        df['MitoVolumeSkel'] = np.count_nonzero(LabSkel)
        
        K_diag_upslope = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        K_diag_downslope = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        K1 = ndi.generate_binary_structure(2, 1)
        branch = []
        AspectRatio = []
        EndPoints = []
        
        for ii in range(1, np.max(LabSkel)):
            SS = LabSkel == ii
            reg = skimage.measure.regionprops(SS * 1)
            if reg[0].minor_axis_length > 0:
                AspectRatio.append(reg[0].major_axis_length / reg[0].minor_axis_length)
            else:
                AspectRatio.append(reg[0].major_axis_length / 1)
            A_orthog = SS.copy()
            B = ndi.convolve(1 * SS, K_diag_upslope, mode='constant')
            A_orthog[B == 2] = 1
            B = ndi.convolve(1 * SS, K_diag_downslope, mode='constant')
            A_orthog[B == 2] = 1
            B = ndi.convolve(1 * A_orthog, K1, mode='constant')
            image_of_branch_points = B >= 4
            branch.append(skimage.morphology.label(image_of_branch_points).max())
            B = np.zeros_like(SS)
            for k in range(9):
                K = np.zeros((9))
                K[4] = 1
                K[k] = 1
                B = B + ndi.convolve(1 * SS, K.reshape(3, 3), mode='constant')
            EndPoints.append(np.count_nonzero(B == 10))

    if branch != []:
        df['MitoMeanBranch'] = np.nanmean(branch)
        df['MitoStdBranchN'] = np.nanstd(branch)
        df['MitoUquanBranchN'] = np.nanquantile(branch, 0.75)
        df['MitoLQuanBranchN'] = np.nanquantile(branch, 0.25)
        df['MitoMedianBranchN'] = np.nanmedian(branch)
    else:
        df['MitoMeanBranch'] = 0
        df['MitoStdBranchN'] = 0
        df['MitoUquanBranchN'] = 0
        df['MitoLQuanBranchN'] = 0
        df['MitoMedianBranchN'] = 0

    if AspectRatio!=[] or EndPoints!=[]:     
        df['MitoMeanAspectRatio'] = np.nanmean(AspectRatio)
        df['MitoStdAspectRatio'] = np.nanstd(AspectRatio)
        df['MitoUquanAspectRatio'] = np.nanquantile(AspectRatio, 0.75)
        df['MitoLQuanAspectRatio'] = np.nanquantile(AspectRatio, 0.25)
        df['MitoMedianAspectRatio'] = np.nanmedian(AspectRatio)
        df['MitoMeanEndPoints'] = np.nanmean(EndPoints)
        df['MitoStdEndPointsN'] = np.nanstd(EndPoints)
        df['MitoUquanEndPointsN'] = np.nanquantile(EndPoints, 0.75)
        df['MitoLQuanEndPointsN'] = np.nanquantile(EndPoints, 0.25)
        df['MitoMedianEndPointsN'] = np.nanmedian(EndPoints)
        if viz==True:
            utils.show_cells([Mseg,skel],title=['Mito','skeleton'])
    else:

        df['MitoMeanAspectRatio'] = 0
        df['MitoStdAspectRatio'] = 0
        df['MitoUquanAspectRatio'] = 0
        df['MitoLQuanAspectRatio'] = 0
        df['MitoMedianAspectRatio'] = 0
        df['MitoMeanEndPoints'] = 0
        df['MitoStdEndPointsN'] = 0
        df['MitoUquanEndPointsN'] = 0
        df['MitoLQuanEndPointsN'] = 0
        df['MitoMedianEndPointsN'] = 0
    return df

def RNA_measurement(Lab,simg,viz=False):
    df=pd.DataFrame([[]])
        
   
    Rn = ndi.gaussian_filter((simg), 1)
    filter_Rn = ndi.gaussian_filter(Rn, 1)
    alpha = 30
    Rn = Rn + alpha * (Rn - filter_Rn)
    Rn=Rn*ndi.binary_erosion(Lab, skimage.morphology.disk(3))
    Rn = Rn > skimage.filters.threshold_otsu(Rn[Rn>1]) * 1.1
    Rn = ndi.binary_opening(Rn, skimage.morphology.disk(1))
    Rn = ndi.binary_closing(Rn)
    for u in range(1, np.max(Rn)):
        if np.nansum(Rn == u) < 5:
            Rn[Rn == u] = 0
    
    Rn = skimage.measure.label(Rn)
    if np.count_nonzero(Rn) < 1:
        
        df['RNACount'] = 0
        df['RNAVolumeMean'] = 0
        df['RNAMeanDistance'] = 0
        df['RNAStdDistance'] = 0
        df['RNAUquantDistance'] = 0
        df['RNALquanrDistance'] = 0
        df['RNAMedianDistance'] = 0
        return df
    if viz==True:
        utils.show_cells([Rn],title=['RNA'])
    
    if Rn.max() > 0:
        df['RNACount'] = np.max(Rn)
        df['RNAVolumeMean'] = np.count_nonzero(Rn) / np.max(Rn)
        distRNA = []
        for tt in range(1, np.max(Rn) + 1):
            for t in range(tt, np.max(Rn) + 1):
                distRNA.append(sp.spatial.distance.pdist([ndi.measurements.center_of_mass(1 * (Rn == tt)),
                                                        ndi.measurements.center_of_mass(1 * (Rn == t))]))
        df['RNAMeanDistance'] = np.nanmean(distRNA)
        df['RNAStdDistance'] = np.nanstd(distRNA)
        df['RNAUquantDistance'] = np.nanquantile(distRNA, 0.75)
        df['RNALquanrDistance'] = np.nanquantile(distRNA, 0.25)
        df['RNAMedianDistance'] = np.nanmedian(distRNA)
    else:
        df['RNACount'] = 0
        df['RNAVolumeMean'] = 0
        df['RNAMeanDistance'] = 0
        df['RNAStdDistance'] = 0
        df['RNAUquantDistance'] = 0
        df['RNALquanrDistance'] = 0
        df['RNAMedianDistance'] = 0
    return df

def Correlation_measurements(simgi, simgj,chan,chanj,Labi,Labj):
    df=pd.DataFrame([[]])
    Cor = np.nanmean(np.corrcoef(simgi, simgj))
    df['Correlation_%s' % chan + '_' + chanj] = Cor
    slope = sp.stats.linregress(simgi.reshape(-1, ), simgj.reshape(-1, ))  # slope value= slope.slope
    df['Correlation_Slope_%s' % chan + '_' + chanj] = slope.slope
    overlap_coeff = np.nansum(simgi * simgj) / np.sqrt(np.nansum(simgi * simgi) * np.nansum(simgj * simgj))
    df['Correlation_Overlap_%s' % chan + '_' + chanj] = overlap_coeff
    M1 = np.nanmean(sum(simgi * Labj) / sum(simgi))
    M2 = np.nanmean(sum(simgj * Labi) / sum(simgj))
    df['Correlation_Mander1_%s' % chan + '_' + chanj] = M1
    df['Correlation_Mander2_%s' % chan + '_' + chanj] = M2
    Rmax = np.max([len(np.unique(simgi)), len(np.unique(simgj))])  # Aux
    Di = abs(len(np.unique(simgi)) - len(np.unique(simgj)))  # Aux
    Wi = (Rmax - Di) / Rmax  # Aux
    RWC1 = sum(sum(simgi * Labj * Wi) / sum(simgi))
    RWC2 = sum(sum(simgj * Labi * Wi) / sum(simgj))
    df['Correlation_RWC1_%s' % chan + '_' + chanj] = RWC1
    df['Correlation_RWC2_%s' % chan + '_' + chanj] = RWC2
    
    return df