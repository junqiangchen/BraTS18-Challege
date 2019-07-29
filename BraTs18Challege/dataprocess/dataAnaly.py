from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path

bratshgg_path = "D:\Data\\brats18\HGG"
bratslgg_path = "D:\Data\\brats18\LGG"
flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"


def getImageSizeandSpacing():
    """
    get image and spacing
    :return:
    """
    pathhgg_list = file_name_path(bratshgg_path)
    pathlgg_list = file_name_path(bratslgg_path)
    for subsetindex in range(len(pathhgg_list)):
        brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        print('subsetindex:', subsetindex)
        print("flair_src image size,flair_src image Spacing:", (flair_src.GetSize(), flair_src.GetSpacing()))
        print("t1_src image size,t1_src image Spacing:", (t1_src.GetSize(), t1_src.GetSpacing()))
        print("t1ce_src image size,t1ce_src image Spacing:", (t1ce_src.GetSize(), t1ce_src.GetSpacing()))
        print("t2_src image size,t2_src image Spacing:", (t2_src.GetSize(), t2_src.GetSpacing()))
    for subsetindex in range(len(pathlgg_list)):
        brats_subset_path = bratslgg_path + "/" + str(pathlgg_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(pathlgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t2_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        print('subsetindex:', subsetindex)
        print("flair_src image size,flair_src image Spacing:", (flair_src.GetSize(), flair_src.GetSpacing()))
        print("t1_src image size,t1_src image Spacing:", (t1_src.GetSize(), t1_src.GetSpacing()))
        print("t1ce_src image size,t1ce_src image Spacing:", (t1ce_src.GetSize(), t1ce_src.GetSpacing()))
        print("t2_src image size,t2_src image Spacing:", (t2_src.GetSize(), t2_src.GetSpacing()))


def getMaskLabelValue():
    """
    get max mask value
    brats mask have four value:0,1,2,4(0 is backgroud )
    :return:
    """
    pathhgg_list = file_name_path(bratshgg_path)
    pathlgg_list = file_name_path(bratslgg_path)
    for subsetindex in range(len(pathhgg_list)):
        brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
        mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
        seg = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        segimg = sitk.GetArrayFromImage(seg)
        seg_maskimage = segimg.copy()
        seg_maskimage = seg_maskimage.flatten()
        bcounts = np.bincount(seg_maskimage)
        print("mask_value:", bcounts)
    print('lgg')
    for subsetindex in range(len(pathlgg_list)):
        brats_subset_path = bratslgg_path + "/" + str(pathlgg_list[subsetindex]) + "/"
        mask_image = brats_subset_path + str(pathlgg_list[subsetindex]) + mask_name
        seg = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        segimg = sitk.GetArrayFromImage(seg)
        seg_maskimage = segimg.copy()
        seg_maskimage = seg_maskimage.flatten()
        bcounts = np.bincount(seg_maskimage)
        print("mask_value:", bcounts)

# getMaskLabelValue()
# getImageSizeandSpacing()
