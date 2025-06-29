# AC-VC-paper
This is code to plot the figures in Egea-Weiss*,Turner-Bridger* et al. (####).  
The corresponding data is available at: ###

## Installation
First create a conda environment with relevant packages

> conda create -n paperFigures python=3.9.12 numpy=1.22.3 pandas=1.5.3 scipy=1.7.3 matplotlib=3.5.1 imageio=2.9.0 statsmodels=0.13 scikit-learn=1.0.2 jupyter seaborn  

Activate environment  

> conda activate paperFigures  

Then, run these lines to install some extra packages  

> pip install tqdm pynrrd mat73 opencv-python==4.7.0.68  
> pip install allensdk

You will need to install the matlab engine for python. The exact version to install depends on the versions of matlab and python on your system. See https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html for more information  

> python -m pip install matlabengine==9.13.1

Lastly, you will need to clone the ccf_streamlines repository, available at: https://github.com/AllenInstitute/ccf_streamlines.git  

## How to use
Figures are plotted in the jupyter notebooks, which include "main" in their title. They rely on functions defined in the respective python modules. Most figures and supplementary figures have their own jupyter notebook, with the expection os S8, S9 and S10, which are plotted within the notebooks for main Figures 3 and 4. 

