from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d_multilabel import Vnet3dModuleMultiLabel
from Vnet.util import getLargestConnectedCompont
from dataprocess.utils import calcu_dice, file_name_path
from dataprocess.data3dprepare import normalize
import numpy as np
import SimpleITK as sitk


def inference():
    """
    Vnet network segmentation brats fine segmatation
    :return:
    """
    channel = 4
    numclass = 4
    flair_name = "_flair.nii.gz"
    t1_name = "_t1.nii.gz"
    t1ce_name = "_t1ce.nii.gz"
    t2_name = "_t2.nii.gz"
    mask_name = "_seg.nii.gz"
    out_mask_name = "_outseg.nii.gz"
    # step1 init vnet model
    depth_z = 48
    Vnet3d = Vnet3dModuleMultiLabel(240, 240, depth_z, channels=channel, numclass=numclass,
                                    costname=("categorical_dice",), inference=True,
                                    model_path="log\segmeation2mm\weighted_categorical_crossentropy\model\Vnet3d.pd-10000")
    brats_path = "D:\Data\\brats18\\test"
    # step2 get all test image path
    dice_values0 = []
    dice_values1 = []
    dice_values2 = []
    dice_values3 = []
    path_list = file_name_path(brats_path)
    # step3 get test image(4 model) and mask
    for subsetindex in range(len(path_list)):
        # step4 load test image(4 model) and mask as ndarray
        brats_subset_path = brats_path + "/" + str(path_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(path_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(path_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(path_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(path_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(path_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        label = sitk.GetArrayFromImage(mask)
        # step5 mormazalation test image(4 model) and merage to 4 channels ndarray
        flair_array = normalize(flair_array)
        t1_array = normalize(t1_array)
        t1ce_array = normalize(t1ce_array)
        t2_array = normalize(t2_array)

        imagez, height, width = np.shape(flair_array)[0], np.shape(flair_array)[1], np.shape(flair_array)[2]
        fourmodelimagearray = np.zeros((imagez, height, width, channel), np.float)
        fourmodelimagearray[:, :, :, 0] = flair_array
        fourmodelimagearray[:, :, :, 1] = t1_array
        fourmodelimagearray[:, :, :, 2] = t1ce_array
        fourmodelimagearray[:, :, :, 3] = t2_array
        ys_pd_array = np.zeros((imagez, height, width), np.uint8)
        # step6 predict test image(4 model)
        last_depth = 0
        for depth in range(0, imagez // depth_z, 1):
            patch_xs = fourmodelimagearray[depth * depth_z:(depth + 1) * depth_z, :, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[depth * depth_z:(depth + 1) * depth_z, :, :] = pathc_pd
            last_depth = depth
        if imagez != depth_z * last_depth:
            patch_xs = fourmodelimagearray[(imagez - depth_z):imagez, :, :, :]
            pathc_pd = Vnet3d.prediction(patch_xs)
            ys_pd_array[(imagez - depth_z):imagez, :, :] = pathc_pd

        ys_pd_array = np.clip(ys_pd_array, 0, 255).astype('uint8')
        all_ys_pd_array = ys_pd_array.copy()
        all_ys_pd_array[ys_pd_array != 0] = 1
        outmask = getLargestConnectedCompont(sitk.GetImageFromArray(all_ys_pd_array))
        ys_pd_array[outmask == 0] = 0
        # step7 calcu test mask and predict mask dice value
        batch_ys = label.copy()
        batch_ys[label == 4] = 3
        dice_value0 = 0
        dice_value1 = 0
        dice_value2 = 0
        dice_value3 = 0
        for num_class in range(4):
            ys_pd_array_tmp = ys_pd_array.copy()
            batch_ys_tmp = batch_ys.copy()
            ys_pd_array_tmp[ys_pd_array == num_class] = 1
            batch_ys_tmp[label == num_class] = 1
            if num_class == 0:
                dice_value0 = calcu_dice(ys_pd_array_tmp, batch_ys_tmp, 1)
            if num_class == 1:
                dice_value1 = calcu_dice(ys_pd_array_tmp, batch_ys_tmp, 1)
            if num_class == 2:
                dice_value2 = calcu_dice(ys_pd_array_tmp, batch_ys_tmp, 1)
            if num_class == 3:
                dice_value3 = calcu_dice(ys_pd_array_tmp, batch_ys_tmp, 1)
        print("index,dice:", (subsetindex, dice_value0, dice_value1, dice_value2, dice_value3))
        dice_values0.append(dice_value0)
        dice_values1.append(dice_value1)
        dice_values2.append(dice_value2)
        dice_values3.append(dice_value3)
        # step8 out put predict mask
        ys_pd_array = ys_pd_array.astype('float')
        outputmask = np.zeros((imagez, height, width), np.uint8)
        outputmask[ys_pd_array == 1] = 1
        outputmask[ys_pd_array == 2] = 2
        outputmask[ys_pd_array == 3] = 4
        ys_pd_itk = sitk.GetImageFromArray(outputmask)
        ys_pd_itk.SetSpacing(mask.GetSpacing())
        ys_pd_itk.SetOrigin(mask.GetOrigin())
        ys_pd_itk.SetDirection(mask.GetDirection())
        out_mask_image = brats_subset_path + str(path_list[subsetindex]) + out_mask_name
        sitk.WriteImage(ys_pd_itk, out_mask_image)
    average0 = sum(dice_values0) / len(dice_values0)
    average1 = sum(dice_values1) / len(dice_values1)
    average2 = sum(dice_values2) / len(dice_values2)
    average3 = sum(dice_values3) / len(dice_values3)
    print("average dice:", (average0, average1, average2, average3))


inference()
