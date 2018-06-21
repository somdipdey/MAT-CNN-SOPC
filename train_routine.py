from engine.bottleneck_features import retrain_classifier


# one of `VGG16`, `VGG19`, `ResNet50`, `VGG16_Places365`


model, elapsed_time = retrain_classifier(pre_trained_model='VGG16',
                                         pooling_mode='avg',
                                         classes=4,
                                         data_augm_enabled = False)





