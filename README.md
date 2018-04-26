# CaImAn_DendriteAnalysis
Jupyter notebooks used in the Helmchen lab for analysis of dendritic two-photon calcium imaging data. The notebooks rely on the the [CaImAn](https://github.com/flatironinstitute/CaImAn) toolbox.

Developed by Henry Luetcke, [ETH Scientific IT Service](https://sis.id.ethz.ch/).

## Workflow overview

The workflow consists of two notebooks for [motion correction](https://github.com/HelmchenLabSoftware/CaImAn_DendriteAnalysis/blob/master/01_Preprocess_MotionCorrect.ipynb) and [source extraction](https://github.com/HelmchenLabSoftware/CaImAn_DendriteAnalysis/blob/master/02_SourceExtract.ipynb). 

### Motion correction: [01_Preprocess_MotionCorrect.ipynb](https://github.com/HelmchenLabSoftware/CaImAn_DendriteAnalysis/blob/master/01_Preprocess_MotionCorrect.ipynb)
1. Select data folder (date, session) and number of trials to process
2. Crop TIF files (optional; to remove artefacts at the borders)
3. Join single-trial TIF files into a single, joined TIF file 
    - this step also saves a metadata file with info about contained trials in JSON format
4. Run motion correction on joined TIF file
    - both rigid and pw-rigid motion correction and run and the results are saved as mmap and TIF files
5. Assess quality of motion correction and residual motion
    - also detect frames with bad residual motion (store info in JSON file)

### Source extraction: [02_SourceExtract.ipynb](https://github.com/HelmchenLabSoftware/CaImAn_DendriteAnalysis/blob/master/02_SourceExtract.ipynb)
1. Select the mmap file created during motion correction (either rigid or pw-rigid correction) and (optionally) remove bad frames
2. Run CNMF algorithm on the joined mmap file (single run, full FoV, no patches)
3. Estimate quality of components and classify as good / bad
4. Calculate DFF and plot traces
5. Split up traces by trials and save as .mat file for further analysis in Matlab
