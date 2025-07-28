# AC-VC-paper
This is code to plot the figures in Egea-Weiss*,Turner-Bridger* et al.(2025)  
The corresponding data will be made available after peer-review

## Installation
First create a conda environment

`conda create -n paperFigures python=3.9.12`

Activate environment  

`conda activate paperFigures` 

Navigate to the package folder then, run the following line to install some the relevant packages into your conda enviroment

`pip install -e.`

You will need to install the matlab engine for python. The exact version to install depends on the versions of matlab and python on your system. See https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html for more information  

`python -m pip install matlabengine==9.13.1`

Lastly, you will need to clone the ccf_streamlines repository, available at: https://github.com/AllenInstitute/ccf_streamlines.git  

## How to use
Figures are plotted in the jupyter notebooks, which include **"main"** in their title. They rely on functions defined in the respective python modules. Most figures and supplementary figures have their own jupyter notebook, with the exception of S8, S9 and S10, which are plotted within the notebooks for main Figures 3 and 4.
Linear mixed models for in vivo analysis are implemented in MATLAB, with functions defined in the MATLAB scripts. 

For reproducing figures involving MAPseq datasets in Figs 1 and 2, you need to adjust the proj_path in the `general_analysis_parameters.yaml` to point to the the path to the processed_A1_MAPseq_datasets folder:
`proj_path: /path/to/your/processed_A1_MAPseq_datasets`

In addition, to generate figures involving flatmaps, download the following data files from ccf_streamlines (https://ccf-streamlines.readthedocs.io/en/latest/data_files.html):
-annotation_25.nrrd
-flatmap_butterfly.h5
-flatmap_butterfly.nrrd
-labelDescription_ITKSNAPColor.txt
-surface_paths_10_v3.h5

Then update the following field in the `general_analysis_parameters.yaml`:
`path_to_additional_req: /path/to/your/additional_data_files`

