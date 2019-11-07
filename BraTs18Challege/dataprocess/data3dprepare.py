from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
from dataprocess.utils import file_name_path

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"


def subimage_generator(image, mask, patch_block_size, numberxy, numberz):
    """
    generate the sub images and masks with patch_block_size
    :param image:
    :param patch_block_size:
    :param stride:
    :return:
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    stridewidth = (width - block_width) // numberxy
    strideheight = (height - block_height) // numberxy
    stridez = (imagez - blockz) // numberz
    # step 1:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
    if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2
        hr_samples_list = []
        hr_mask_samples_list = []
        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
                for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
                    if np.max(mask[z:z + blockz, x:x + block_width, y:y + block_height]) != 0:
                        hr_samples_list.append(image[z:z + blockz, x:x + block_width, y:y + block_height])
                        hr_mask_samples_list.append(mask[z:z + blockz, x:x + block_width, y:y + block_height])
        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, block_width, block_height))
        hr_mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        return hr_samples, hr_mask_samples
    # step 2:other sutitation,return one samples
    else:
        nb_sub_images = 1 * 1 * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float)
        rangz = lambda imagez, blockz: imagez if imagez < blockz else blockz
        rangwidth = lambda width, block_width: width if width < block_width else block_width
        rangheight = lambda height, block_height: height if width < block_height else block_height
        hr_samples[0, 0:blockz, 0:block_width, 0:block_height] = image[0:rangz, 0:rangwidth, 0:rangheight]
        hr_mask_samples[0, 0:blockz, 0:block_width, 0:block_height] = mask[0:rangz, 0:rangwidth, 0:rangheight]
        return hr_samples, hr_mask_samples


def make_patch(image, mask, patch_block_size, numberxy, numberz):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
    """
    image_subsample, mask_subsample = subimage_generator(image=image, mask=mask, patch_block_size=patch_block_size,
                                                         numberxy=numberxy, numberz=numberz)
    return image_subsample, mask_subsample


def gen_image_mask(flairimg, t1img, t1ceimg, t2img, segimg, index, shape, numberxy, numberz, trainImage, trainMask,
                   part):
    """
    :param flairimg:
    :param t1img:
    :param t1ceimg:
    :param t2img:
    :param segimg:
    :param index:
    :param shape:
    :param numberxy:
    :param numberz:
    :param trainImage:
    :param trainMask:
    :param part:
    :return:
    """
    # step 2 get subimages (numberxy*numberxy*numberz,64, 128, 128)
    sub_flairimages,sub_maskimages = make_patch(flairimg,segimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    sub_t1images,_ = make_patch(t1img,segimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    sub_t1ceimages,_ = make_patch(t1ceimg,segimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    sub_t2images,_ = make_patch(t2img,segimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    # step 3 only save subimages (numberxy*numberxy*numberz,64, 128, 128)
    samples, imagez, height, width = np.shape(sub_flairimages)[0], np.shape(sub_flairimages)[1], \
                                     np.shape(sub_flairimages)[2], np.shape(sub_flairimages)[3]
    for j in range(samples):
        sub_masks = sub_maskimages.astype(np.float)
        sub_masks = np.clip(sub_masks, 0, 255).astype('uint8')
        if np.max(sub_masks[j, :, :, :]) != 0:
            """
            merage 4 model image into 4 channel (imagez,width,height,channel)
            """
            fourmodelimagearray = np.zeros((imagez, height, width, 4), np.float)
            filepath1 = trainImage + "\\" + str(part) + "_" + str(index) + "_" + str(j) + ".npy"
            filepath = trainMask + "\\" + str(part) + "_" + str(index) + "_" + str(j) + ".npy"
            flairimage = sub_flairimages[j, :, :, :]
            flairimage = flairimage.astype(np.float)
            fourmodelimagearray[:, :, :, 0] = flairimage
            t1image = sub_t1images[j, :, :, :]
            t1image = t1image.astype(np.float)
            fourmodelimagearray[:, :, :, 1] = t1image
            t1ceimage = sub_t1ceimages[j, :, :, :]
            t1ceimage = t1ceimage.astype(np.float)
            fourmodelimagearray[:, :, :, 2] = t1ceimage
            t2image = sub_t2images[j, :, :, :]
            t2image = t2image.astype(np.float)
            fourmodelimagearray[:, :, :, 3] = t2image
            np.save(filepath1, fourmodelimagearray)
            np.save(filepath, sub_masks[j, :, :, :])


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp


def prepare3dtraindata(pathhgg_list, bratshgg_path, trainImage, trainMask, shape=(16, 256, 256), numberxy=3,
                       numberz=20, part=1):
    """
    :param pathhgg_list:
    :param bratshgg_path:
    :param trainImage:
    :param trainMask:
    :param shape:
    :param numberxy:
    :param numberz:
    :return:
    """
    """
    load flair_image,t1_image,t1ce_image,t2_image,mask_image
    """
    for subsetindex in range(len(pathhgg_list)):
        brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        """
        normalize every model image,mask is not normalization
        """
        flair_array = normalize(flair_array)
        t1_array = normalize(t1_array)
        t1ce_array = normalize(t1ce_array)
        t2_array = normalize(t2_array)

        gen_image_mask(flair_array, t1_array, t1ce_array, t2_array, mask_array, subsetindex, shape=shape,
                       numberxy=numberxy, numberz=numberz, trainImage=trainImage, trainMask=trainMask, part=part)


def preparetraindata():
    """
    :return:
    """
    bratshgg_path = "D:\Data\\brats18\HGG"
    bratslgg_path = "D:\Data\\brats18\LGG"
    trainImage = "D:\Data\\brats18\\train\Image"
    trainMask = "D:\Data\\brats18\\train\Mask"
    pathhgg_list = file_name_path(bratshgg_path)
    pathlgg_list = file_name_path(bratslgg_path)

    prepare3dtraindata(pathhgg_list, bratshgg_path, trainImage, trainMask, (64, 128, 128), 3, 15, 1)
    prepare3dtraindata(pathlgg_list, bratslgg_path, trainImage, trainMask, (64, 128, 128), 3, 15, 2)


#preparetraindata()
