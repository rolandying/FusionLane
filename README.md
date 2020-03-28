# FusionLane: 
This is the source code of paper [*FusionLane: Multi-Sensor Fusion for Lane Marking Semantic Segmentation Using Deep Neural Networks*](https://arxiv.org/abs/2003.04404). The source code of this project is modified based on [deeplabV3+](https://github.com/rishizek/tensorflow-deeplab-v3-plus).

## Setup
Requirements:

* Tensorflow>= 1.12
* Numpy
* matplotlib
* opencv-python

## Dataset
This project uses the TFRecord format to consume data in the training and evaluation process. The TFRecord used in this project could is available at [here](https://drive.google.com/open?id=1wOLO--uDOpd6jECevHjf07-WMx_FEf_e "Tfrecord"). You can also find the testing images at [here](https://drive.google.com/open?id=15IKQ4eVhmbV7J3Ibu1PNXgC89gMnZvQP).

## Training and Evaluation
You can start your own training from scratch as follows. Or you could just modified in the `train.py` file.
``` 
python  train.py -- data_dir TFRECORD_DATA_DIR \
                 -- model_dir MODEL_DIR
```
You can evaluate you model in the same way as the training. Our model is available at [here](https://drive.google.com/open?id=1Cab7cuS_HfSpzQGR5SrWdyGZz5oxsgvl).

-----
The rests are still under construction... O(∩_∩)O
