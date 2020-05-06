from __future__ import print_function, division
import tensorflow as tf
import numpy as np


# --------------------------- BINARY Evaluation ---------------------------
def binary_iou(Y_pred, Y_gt, prob=0.5):
    """
    binary iou
    :param Y_pred:A tensor resulting from a sigmod
    :param Y_gt:A tensor of the same shape as `output`
    :return: binary iou
    """
    Y_pred_part = tf.to_float(Y_pred > prob)
    Y_pred_part = tf.cast(Y_pred_part, tf.float32)
    Y_gt_part = tf.identity(Y_gt)
    Y_gt_part = tf.cast(Y_gt_part, tf.float32)
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    pred_flat = tf.reshape(Y_pred_part, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt_part, [-1, H * W * C * Z])
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=-1)
    union = tf.reduce_sum(pred_flat, axis=-1) + tf.reduce_sum(true_flat, axis=-1) - intersection
    metric = tf.reduce_mean((intersection + smooth_tf) / (union + smooth_tf))
    metric = tf.cond(tf.is_inf(metric), lambda: smooth_tf, lambda: metric)
    return metric


# --------------------------- BINARY LOSSES ---------------------------
def binary_crossentropy(Y_pred, Y_gt):
    """
    Binary crossentropy between an output tensor and a target tensor.
    :param Y_pred:A tensor with (batchsize,z,x,y,channel)from a sigmod,probability distribution.
    :param Y_gt:A tensor with the same shape as `output`.
    :return:binary_crossentropy
    """
    epsilon = 1.e-5
    Y_pred = tf.clip_by_value(Y_pred, epsilon, 1. - epsilon)
    logits = tf.log(Y_pred / (1 - Y_pred))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_gt, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss


def weighted_binary_crossentroy(Y_pred, Y_gt, beta):
    """
    Weighted cross entropy (WCE) is a variant of CE where all positive examples get weighted by some coefficient.
    It is used in the case of class imbalance.
    For example, when you have an image with 10% black pixels and 90% white pixels, regular CE won’t work very well.
    WCE define:wce(p',p)=-(b*p*log(p')+(1-p)*log(1-p'))
    :param Y_pred:A tensor with (batchsize,z,x,y,channel)from a sigmod,probability distribution.
    :param Y_gt:A tensor with the same shape as `output`.
    :param beta: To decrease the number of false negatives, setβ>1. To decrease the number of false positives, set β<1.
    :return:weighted_binary_crossentroy
    """
    epsilon = 1.e-5
    Y_pred = tf.clip_by_value(Y_pred, epsilon, 1. - epsilon)
    logits = tf.log(Y_pred / (1 - Y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=Y_gt, logits=logits, pos_weight=beta)
    loss = tf.reduce_mean(loss)
    return loss


def balanced_binary_crossentroy(Y_pred, Y_gt, beta):
    """
    Balanced cross entropy (BCE) is similar to WCE. The only difference is that we weight also the negative examples
    bce define:bce(p',p)=-(b*p*log(p')+(1-b)*(1-p)*log(1-p'))
    :param Y_pred:A tensor with (batchsize,z,x,y,channel)from a sigmod,probability distribution.
    :param Y_gt:A tensor with the same shape as `output`.
    :param beta: β!=1,the denominator in pos_weight is not defined
    such as：beta = tf.reduce_sum(1 - y_true) / (BATCH_SIZE * HEIGHT * WIDTH)
    :return:
    """
    epsilon = 1.e-5
    Y_pred = tf.clip_by_value(Y_pred, epsilon, 1. - epsilon)
    logits = tf.log(Y_pred / (1 - Y_pred))
    beta = tf.clip_by_value(beta, epsilon, 1 - epsilon)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=Y_gt, logits=logits, pos_weight=pos_weight)
    loss = tf.reduce_mean(loss * (1 - beta))
    return loss


def binary_dice(Y_pred, Y_gt):
    """
    binary dice loss
    loss=2*(p&p')/(p+p')
    :param Y_pred: A tensor resulting from a sigmod
    :param Y_gt:  A tensor of the same shape as `output`
    :return: binary dice loss
    """
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    pred_flat = tf.cast(Y_pred, tf.float32)
    true_flat = tf.cast(Y_gt, tf.float32)
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=-1) + smooth_tf
    denominator = tf.reduce_sum(pred_flat, axis=-1) + tf.reduce_sum(true_flat, axis=-1) + smooth_tf
    loss = -tf.reduce_mean(intersection / denominator)
    return loss


def binary_tversky(Y_pred, Y_gt, beta):
    """
    Tversky loss (TL) is a generalization of Dice loss. TL adds a weight to FP and FN.
    define:TL(p,p')=(p&p')/(p&p'+b*((1-p)&p')+(1-b)*(p&(1-p')))
    :param Y_pred:A tensor resulting from a sigmod
    :param Y_gt:A tensor of the same shape as `output`
    :param beta:beta=1/2,just Dice loss,beta must(0,1)
    :return:
    """
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    pred_flat = tf.cast(Y_pred, tf.float32)
    true_flat = tf.cast(Y_gt, tf.float32)
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=-1) + smooth_tf
    denominator = intersection + tf.reduce_sum(beta * pred_flat * (1 - true_flat), axis=-1) + \
                  tf.reduce_sum((1 - beta) * true_flat * (1 - pred_flat), axis=-1) + smooth_tf
    loss = -tf.reduce_mean(intersection / denominator)
    return loss


def binary_dicePcrossentroy(Y_pred, Y_gt, lamda=1):
    """
    plus dice and crossentroy loss
    :param Y_pred:A tensor resulting from a sigmod
    :param Y_gt:A tensor of the same shape as `output`
    :param lamda:can set 0.1,0.5,1
    :return:dice+crossentroy
    """
    # step 1,calculate binary crossentroy
    loss1 = binary_crossentropy(Y_pred, Y_gt)
    # step 2,calculate binary dice
    loss2 = 1 - binary_dice(Y_pred, Y_gt)
    # step 3,calculate all loss mean
    loss = lamda * loss1 + tf.log1p(loss2)
    return loss


def binary_focalloss(Y_pred, Y_gt, alpha=0.25, gamma=2.):
    """
    Binary focal loss.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param Y_pred: A tensor resulting from a sigmoid
    :param alpha: Sample category weight
    :param gamma: Difficult sample weight
    :return: Binary focal loss.
    """
    epsilon = 1.e-5
    pt_1 = tf.where(tf.equal(Y_gt, 1), Y_pred, tf.ones_like(Y_pred))
    pt_0 = tf.where(tf.equal(Y_gt, 0), Y_pred, tf.zeros_like(Y_pred))
    # clip to prevent NaN's and Inf's
    pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
    pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)
    loss_1 = alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)
    loss_0 = (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0)
    loss = -tf.reduce_sum(loss_1 + loss_0)
    loss = tf.reduce_mean(loss)
    return loss


def binary_distanceloss(Y_pred, Y_gt):
    """
    distance loss,make segmentation network more sensitive to the boundaries
    can use with cross entroy ,dice together,but should have mutilfy weighting coefficient
    :param Y_pred:A tensor resulting from a sigmoid
    :param Y_gt:A tensor of the same shape as `y_pred`
    :return:
    """
    pred_flat = tf.cast(Y_pred, tf.float32)
    true_flat = tf.cast(Y_gt, tf.float32)

    def Edge_Extracted(y_pred):
        # Edge extracted by sobel filter
        min_x = tf.constant(0, tf.float32)
        max_x = tf.constant(1, tf.float32)
        sobel_x = tf.constant([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], tf.float32)
        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 3, 1, 1])
        sobel_y_filter = tf.transpose(sobel_x_filter, [0, 2, 1, 3, 4])
        filters_x = tf.nn.conv3d(y_pred, sobel_x_filter, strides=[1, 1, 1, 1, 1], padding='SAME')
        filters_y = tf.nn.conv3d(y_pred, sobel_y_filter, strides=[1, 1, 1, 1, 1], padding='SAME')
        edge = tf.sqrt(filters_x * filters_x + filters_y * filters_y + 1e-16)
        edge = tf.clip_by_value(edge, min_x, max_x)
        return edge

    Y_pred_edge = Edge_Extracted(pred_flat)
    Y_gt_edge = Edge_Extracted(true_flat)
    distanceloss = tf.reduce_sum(Y_gt_edge * Y_pred_edge, axis=-1)
    return distanceloss


# --------------------------- MULTICLASS Evaluation ---------------------------

def mean_iou(Y_pred, Y_gt):
    """
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes,but label 0 is background,general background is more big,
    so mean iou calculate don't include background
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :return: mean_iou
    """
    num_class = Y_pred.get_shape().as_list()[-1]
    Y_pred_part = tf.one_hot(tf.argmax(Y_pred, axis=-1), num_class)
    Y_pred_part = tf.cast(Y_pred_part, tf.float32)
    Y_pred_part = Y_pred_part[:, :, :, :, 1:num_class]
    Y_gt_part = tf.cast(Y_gt, tf.float32)
    Y_gt_part = Y_gt_part[:, :, :, :, 1:num_class]
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    pred_flat = tf.reshape(Y_pred_part, [-1, H * W * Z])
    true_flat = tf.reshape(Y_gt_part, [-1, H * W * Z])
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=-1)
    union = tf.reduce_sum(pred_flat, axis=-1) + tf.reduce_sum(true_flat, axis=-1) - intersection
    metric = tf.reduce_mean((intersection + smooth_tf) / (union + smooth_tf))
    metric = tf.cond(tf.is_inf(metric), lambda: smooth_tf, lambda: metric)
    return metric


def mean_dice(Y_pred, Y_gt):
    """
    Mean dice is a common evaluation metric for
    semantic image segmentation, which first computes the dice for each
    semantic class and then computes the average over classes,but label 0 is background,general background is more big,
    so mean dice calculate don't include background
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :return: mean_iou
    """
    num_class = Y_pred.get_shape().as_list()[-1]
    Y_pred_part = tf.one_hot(tf.argmax(Y_pred, axis=-1), num_class)
    Y_pred_part = tf.cast(Y_pred_part, tf.float32)
    Y_pred_part = Y_pred_part[:, :, :, :, 1:num_class]
    Y_gt_part = tf.cast(Y_gt, tf.float32)
    Y_gt_part = Y_gt_part[:, :, :, :, 1:num_class]
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    pred_flat = tf.reshape(Y_pred_part, [-1, H * W * Z])
    true_flat = tf.reshape(Y_gt_part, [-1, H * W * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=-1)
    union = tf.reduce_sum(pred_flat, axis=-1) + tf.reduce_sum(true_flat, axis=-1)
    metric = tf.reduce_mean((intersection + smooth_tf) / (union + smooth_tf))
    metric = tf.cond(tf.is_inf(metric), lambda: smooth_tf, lambda: metric)
    return metric


# --------------------------- MULTICLASS LOSSES ---------------------------
def categorical_crossentropy(Y_pred, Y_gt):
    """
    Categorical crossentropy between an output and a target
    loss=-y*log(y')
    :param Y_pred: A tensor resulting from a softmax
    :param Y_gt:  A tensor of the same shape as `output`
    :return:categorical_crossentropy loss
    """
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = -Y_gt * tf.log(output)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(loss)
    return loss


def weighted_categorical_crossentropy(Y_pred, Y_gt, weights):
    """
    weighted_categorical_crossentropy between an output and a target
    loss=-weight*y*log(y')
    :param Y_pred:A tensor resulting from a softmax
    :param Y_gt:A tensor of the same shape as `output`
    :param weights:numpy array of shape (C,) where C is the number of classes
    :return:categorical_crossentropy loss
    Usage:
    weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    """
    weights = np.array(weights)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = - Y_gt * tf.log(output)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weights * loss)
    return loss


def categorical_dice(Y_pred, Y_gt, weight_loss):
    """
    multi label dice loss with weighted
    WDL=1-2*(sum(w*sum(r&p))/sum((w*sum(r+p)))),w=array of shape (C,)
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :param weight_loss: numpy array of shape (C,) where C is the number of classes
    :return:
    """
    weight_loss = np.array(weight_loss)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_mean(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss = -tf.reduce_mean(weight_loss * gen_dice_coef)
    return loss


def categorical_tversky(Y_pred, Y_gt, beta, weight_loss):
    """
    multi label tversky with weighted
    Tversky loss (TL) is a generalization of Dice loss. TL adds a weight to FP and FN.
    define:TL(p,p')=(p&p')/(p&p'+b*((1-p)&p')+(1-b)*(p&(1-p')))
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :param beta:beta=1/2,just Dice loss,beta must(0,1)
    :return:
    """
    weight_loss = np.array(weight_loss)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    p0 = Y_pred
    p1 = 1 - Y_pred
    g0 = Y_gt
    g1 = 1 - Y_gt
    # Compute gen dice coef:
    numerator = p0 * g0
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = tf.reduce_sum(beta * p0 * g1, axis=(1, 2, 3)) + tf.reduce_sum((1 - beta) * p1 * g0,
                                                                                axis=(1, 2, 3)) + numerator
    gen_dice_coef = tf.reduce_mean((numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss = -tf.reduce_mean(weight_loss * gen_dice_coef)
    return loss


def generalized_dice_loss_w(Y_pred, Y_gt):
    """
    Generalized Dice Loss with class weights
    GDL=1-2*(sum(w*sum(r*p))/sum((w*sum(r+p)))),w=1/sum(r)*sum(r)
    rln为类别l在第n个像素的标准值(GT)，而pln为相应的预测概率值。此处最关键的是wl,为每个类别的权重
    :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :param Y_pred:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :return:
    """
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    weight_loss = tf.reduce_sum(Y_gt, axis=(0, 1, 2, 3))
    weight_loss = 1 / (tf.pow(weight_loss, 2) + smooth_tf)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = weight_loss * tf.reduce_sum(numerator, axis=(0, 1, 2, 3))
    numerator = tf.reduce_sum(numerator)
    denominator = Y_gt + Y_pred
    denominator = weight_loss * tf.reduce_sum(denominator, axis=(0, 1, 2, 3))
    denominator = tf.reduce_sum(denominator)
    loss = -2 * (numerator + smooth_tf) / (denominator + smooth_tf)
    return loss


def categorical_focal_loss(Y_pred, Y_gt, gamma, alpha):
    """
     Categorical focal_loss between an output and a target
    :param Y_pred: A tensor of the same shape as `y_pred`
    :param Y_gt: A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param alpha: Sample category weight,which is shape (C,) where C is the number of classes
    :param gamma: Difficult sample weight
    :return:
    """
    weight_loss = np.array(alpha)
    epsilon = 1.e-5
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weight_loss * loss)
    return loss


def categorical_dicePcrossentroy(Y_pred, Y_gt, weight, lamda=0.5):
    """
    hybrid loss function from dice loss and crossentroy
    loss=Ldice+lamda*Lfocalloss
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param gamma:Difficult sample weight
    :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
    :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
    :return:diceplusfocalloss
    """
    weight_loss = np.array(weight)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = -Y_gt * tf.log(output)
    loss = tf.reduce_mean(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss2 = tf.reduce_mean(weight_loss * loss)
    total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
    return total_loss


def categorical_dicePfocalloss(Y_pred, Y_gt, alpha, lamda=0.5, gamma=2.):
    """
    hybrid loss function from dice loss and focalloss
    loss=Ldice+lamda*Lfocalloss
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param gamma:Difficult sample weight
    :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
    :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
    :return:dicePfocalloss
    """
    weight_loss = np.array(alpha)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
    epsilon = 1.e-5
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_mean(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss2 = tf.reduce_mean(weight_loss * loss)
    total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
    return total_loss


def ssim2d_loss(Y_pred, Y_gt, maxlabel):
    """
    Computes SSIM index between Y_pred and Y_gt.only calculate 2d image,3d image can use it,but not actual ssim3d
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt:A tensor of the same shape as `y_pred`
    :param maxlabel:maxlabelvalue
    :return:ssim_loss
    """
    loss = tf.image.ssim(Y_pred, Y_gt, maxlabel)
    loss = tf.reduce_mean(loss)
    return loss


def multiscalessim2d_loss(Y_pred, Y_gt, maxlabel, downsampledfactor=4):
    """
    Computes the MS-SSIM between Y_pred and Y_gt.only calculate 2d image,3d image can use it,but not actual multiscalessim3d
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt:A tensor of the same shape as `y_pred`
    :param maxlabel:maxlabelvalue
    :param downsampledfactor:downsample factor depend on input imagesize
    :return:multiscalessim_loss
    """
    if downsampledfactor >= 5:
        _MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    if downsampledfactor == 4:
        _MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363)
    if downsampledfactor == 3:
        _MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001)
    if downsampledfactor == 2:
        _MSSSIM_WEIGHTS = (0.0448, 0.2856)
    if downsampledfactor <= 1:
        _MSSSIM_WEIGHTS = (0.0448)
    loss = tf.image.ssim_multiscale(Y_pred, Y_gt, maxlabel, power_factors=_MSSSIM_WEIGHTS)
    loss = tf.reduce_mean(loss)
    return loss
