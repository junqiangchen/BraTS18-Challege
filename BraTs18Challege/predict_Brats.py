import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d import Vnet3dModule
from dataprocess.utils import calcu_dice
import numpy as np
import pandas as pd


def predict():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data/test.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values

    dice_values = []
    Vnet3d = Vnet3dModule(128, 128, 64, channels=4, numclass=3, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\VNet\model\Vnet3d.pd")
    for index in range(imagedata.shape[0]):
        image_gt = np.load(imagedata[index])
        mask_pd = Vnet3d.prediction(image_gt)
        mask_gt = np.load(maskdata[index])
        mask_gt_255 = mask_gt.copy()
        mask_gt_255[mask_gt == 0] = 0
        mask_gt_255[mask_gt != 0] = 255
        dice_value = calcu_dice(mask_pd, mask_gt_255)
        print("index,dice:", (index, dice_value))
        dice_values.append(dice_value)
    average = sum(dice_values) / len(dice_values)
    print("average dice:", average)


predict()
