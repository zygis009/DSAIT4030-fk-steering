# DSAIT4030-fk-steering
Reproduction of the FK diffusion steering paper for TU Delft DSAIT4030 Generative Modeling course

## Setup
First install the requirements on your system or in a virtual python environment:
```
pip install -r requirements.txt
```

### GenEval setup
For running GenEval evaluation, a Mask2Former object detector is required. Install it by running:
```
./flax_fkd/GenEval/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
```

Afterwards, we need to separately install the mmdetection repo and the packages it uses. Run the following commands to install it:
```
pip install -U openmim
mim install mmengine mmcv-full==1.7.2
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```
Note: installing mmcv-full may take a while.