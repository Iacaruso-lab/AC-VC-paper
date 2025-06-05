# AC-VC-paper
This is code to plot the figures in Egea-Weiss*,Turner-Bridger* et al. (####).  
The corresponding data is available at: ###

## How to use
All figures are plotted from makePaperFigures_main.ipynb, using functions contained in the remaining notebooks.   
To run this code, first create a conda envirnoment with the relevant packages. You will also need the dependencies listed below  

> conda create -n paperFigures python=3.9.12 numpy=1.22.3 pandas=1.5.3 scipy=1.7.3 matplotlib=3.5.1 imageio=2.9.0 statsmodels=0.13 jupyter seaborn


Then, run this line to import some extra dependencies
'''
pip install tqdm pynrrd mat73 opencv-python==4.7.0.68
pip install allensdk
python -m pip install matlabengine==9.13.1
'''

## Dependencies  
- allensdk: https://allensdk.readthedocs.io/en/latest/install.html
- ccf streamlines: https://github.com/AllenInstitute/ccf_streamlines.git
- MATLAB engine for Python: https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
