import os
from .dataset_rgb import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR

def get_training_data(rgb_dir, img_options, target_transform=None):
    if target_transform is not None:
        print('Transform get_traning_data not None')
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, target_transform)

def get_validation_data(rgb_dir, target_transform=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, target_transform)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)


