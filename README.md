# NYSCF_ScaleFEx: a slim and scalable method to extract fixed features at single cell level in High Content Imaging screens

We have developed ScaleFEx, a Python pipeline that extracts multiple generic fixed features at the single cell level that can be deployed across large high-content imaging datasets with low computation requirements. 
This pipeline efficiently and reliably computes features related to shape, size, intensity, texture, granularity as well as correlations between channels. Additionally, it allows the measurement of additional features specifically related to mitochondria and RNA only, as they represent important channels with characteristics worth to be measured on their own. The measured features can be used to not only separate populations of cells using AI tools, but also  highlight the specific interpretable features that differ between populations. The authors used ScaleFEx to identify the phenotypic shifts that multiple cell lines undergo when exposed to different compounds. We used a combination of Recursive Feature Elimination, Logistic regression, correlation analysis and dimensionality reduction representations to narrow down to the most meaningful features that described the drug shifts. Furthermore, we used the best scoring features to extract images of cells for each class closest to the average to visually highlight the phenotypic shifts caused by the drugs. Using this approach we were able to identify features linked to the drug shifts in line with literature, and we could visually validate their involvement in the morphological changes of the cells. 
ScaleFEx can be used as a powerful tool to understand the underlying phenotypes of complex diseases and subtle drug shifts, bringing us a step forward to identify and characterize morphological differences at single cell level of the major diseases of our time.
![Figure1_v0 2](https://github.com/NYSCF/NYSCF_ScaleFEx_release/assets/23292813/496f92e7-ac4a-4b5f-a65f-0d2503bc8968)


**A preprint is available [here] (https://www.biorxiv.org/content/10.1101/2023.07.06.547985v1) **

**Download the downsamples (from 2160x2160 to 540x540) imaging dataset [here](https://nyscfopensource.blob.core.windows.net/scalefex/ScaleFEx.zip).**

**NOTE** that the size of the imaging dataset is 142 GBs

**Download the pre-computed ScaleFEx data [here](https://nyscfopensource.blob.core.windows.net/scalefex/ScaleFex_computed_normalized.csv).**

**NOTE** that the size of the csv containing the computed vector is 13.2 GBs

If interested in the full-sized dataset, please email the authors at bmigliori@nyscf.org

## How to use this repository:
To see an example on how to launch and set ScaleFEx over the entire plate experiment, open **compute_ScaleFEx.py** and change the variables to reflect you paths and experiment type.
From this file you can specify if to run the computation on the clouds (AWS servers) or on premise. 
Details on how to set up and run the code on cloud servers are provided in a separate file named **AWS_computing_documentation**

To visualize the masks of single cell's channels while computing ScaleFEx on a small provided example field, open **Example_notebook_ScaleFEx.ipynb**
    
**NOTE** that since the output is provided in the folder, the function will check if that field has been already calculated and won't re-compute the files. To start a new computation delete the current output

The example field is provided in the "data" folder.
**ScaleFEx_on_prem_class.py** and **ScaleFEx_cloud_class.py** are the main functions called to generate the computations. They call all of the other functions to query the data, do pre processing, segmentation, and finally feature extraction. The first is called when deploying the code on premise, the second when the 

**utils.py** contains the functions used to process and query the data
**nuclei_location_extraction.py** contains the functions to perform segmentation and nuclei coordinate extraction. 

To see an example of the analysis, run **Scale_Fex_analysis.ipynb**. 
**NOTE** that a seed was not fixed for the analysis that rely on randomness, therefore some results might differ slightly from the ones in the manuscript

## How to install the dependancies:
ScaleFEx depends on the use of Anaconda which can be downloaded [here](https://www.anaconda.com/products/distribution)
1. Create and activate a conda environment with a modern python version:
	```
	conda create -n ScaleFEx python=3.8.10 pip
	conda activate ScaleFEx
	```
2. Install the required libraries:
	```
	pip install -r requirements.txt
	```
## Citation:
If you find this repository useful for your research, please consider giving us a star ⭐ and cite our [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.06.547985v1) paper:

	
 	ScaleFExSM: a lightweight and scalable method to extract fixed features from single cells in high-content imaging screens
	Bianca Migliori, Neeloy Bose, Daniel Paull
	bioRxiv 2023.07.06.547985; doi: https://doi.org/10.1101/2023.07.06.547985
	




ScaleFEx℠ Dataset © 2023 by NYSCF is licensed under Business Source License 1.1.
