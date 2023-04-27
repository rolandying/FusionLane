# FusionLane: 
This is the source code of paper [*FusionLane: Multi-Sensor Fusion for Lane Marking Semantic Segmentation Using Deep Neural Networks*](https://ieeexplore.ieee.org/document/9237136). The source code of this project is modified based on [deeplabV3+](https://github.com/rishizek/tensorflow-deeplab-v3-plus).

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
## Citation
If you feel our work is helpfull, please cite as follow:
```
@article{yin2020fusionlane,
  title={Fusionlane: Multi-sensor fusion for lane marking semantic segmentation using deep neural networks},
  author={Yin, Ruochen and Cheng, Yong and Wu, Huapeng and Song, Yuntao and Yu, Biao and Niu, Runxin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={2},
  pages={1543--1553},
  year={2020},
  publisher={IEEE}
}
```
