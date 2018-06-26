# MAT-CNN-FPGA: Traffic Analysis Using CNNs on FPGA

## Motionless Analysis of Traffic Using Convolutional Neural Networks on FPGA: MAT-CNN-FPGA

### Train simple CNN (using Matlab)

Use the vgg16_custom.m in /source/matlab/.

### Train simple CNN (using Keras)

Use the train_routine.py in /source/python/. Varying parameters are :

```python
pre_trained_model='VGG16'
```

Also you can change the number of training epochs inside source/python/engine/bottleneck_features.py

The `base_dir` and `base_dir_trained_models` variables must be adapted accordingly.
