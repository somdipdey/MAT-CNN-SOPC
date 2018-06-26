# Traffic Analysis on FPGA



### Train simple CNN (using Keras)

Use the train_routine.py. Varying parameters are :

```python
pre_trained_model='VGG16'

data_augm_enabled = False
```

Also you can change the number of training epochs inside source/python/engine/bottleneck_features.py

The `base_dir` and `base_dir_trained_models` variables must be adapted accordingly.
