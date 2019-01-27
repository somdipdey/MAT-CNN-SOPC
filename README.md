# MAT-CNN-SOPC: Traffic Analysis Using CNNs on FPGA

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/somdipdey/MAT-CNN-FPGA_Traffic_Analysis_Using_CNNs_On_FPGA/blob/master/LICENSE)

## Motionless Analysis of Traffic Using Convolutional Neural Networks on System-on-a-programmable-chip: MAT-CNN-FPGA

### Train simple CNN (using Matlab)

Use the vgg16_custom.m in /source/matlab/. Execution:

    >> vgg16_custom()

### Train simple CNN (using Keras in Python)

Use the train_routine.py in /source/python/. Varying parameters are :

```python
pre_trained_model='VGG16'
```

Also you can change the number of training epochs inside source/python/engine/bottleneck_features.py

The `base_dir` and `base_dir_trained_models` variables must be adapted accordingly.


### Accuracy Result
<img width="750" alt="Face Detection Using OpenCV and Resource Monitoring" src="https://user-images.githubusercontent.com/8515608/41942318-fbaa5f28-7996-11e8-926c-f9575f12c347.png">

## Citation

MAT-CNN-SOPC is presented at **2018 NASA/ESA Conference on Adaptive Hardware and Systems (AHS 2018)**. Link: https://www.ahs-conf.org 

If you are using the MAT-CNN-SOPC code then please do cite the paper as follows:

    Dey, S., Kalliatakis, G., Saha, S., Singh, A. K., Ehsan, S., & McDonald-Maier, K. (2018, August). MAT-CNN-SOPC: Motionless Analysis of Traffic Using Convolutional Neural Networks on System-On-a-Programmable-Chip. In 2018 NASA/ESA Conference on Adaptive Hardware and Systems (AHS) (pp. 291-298). IEEE.
    
Bib:
    
    @inproceedings{dey2018mat,
    title={MAT-CNN-SOPC: Motionless Analysis of Traffic Using Convolutional Neural Networks on System-On-a-Programmable-Chip},
    author={Dey, Somdip and Kalliatakis, Grigorios and Saha, Sangeet and Singh, Amit Kumar and Ehsan, Shoaib and McDonald-Maier, Klaus},
    booktitle={2018 NASA/ESA Conference on Adaptive Hardware and Systems (AHS)},
    pages={291--298},
    year={2018},
    organization={IEEE}
    }
