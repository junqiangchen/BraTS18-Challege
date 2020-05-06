'''

'''
from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add,
                        weight_xavier_init, bias_variable, save_images)
from Vnet.loss_metric import (categorical_crossentropy, mean_iou, mean_dice, categorical_dice, categorical_focal_loss,
                              generalized_dice_loss_w, categorical_tversky, weighted_categorical_crossentropy,
                              categorical_dicePcrossentroy, categorical_dicePfocalloss, multiscalessim2d_loss,
                              ssim2d_loss)
import tensorflow as tf
import numpy as np
import os


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=20, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=20, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv


def conv_softmax(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.softmax(conv)
        return conv


def _create_conv_net(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=2):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 20), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 20, 20), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 20, 40), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 40, 80), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 80, 160), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 160, 320), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 160, 320), scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 320, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 80, 160), scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat(layer3, deconv2)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 160, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 40, 80), scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 80, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_3')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 20, 40), scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 40, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_3')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_softmax(x=layer9, kernal=(1, 1, 1, 20, n_class), scope='output')

    return output_map


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


# convet label to one hot type
def convert_to_one_hot(y, numclass):
    """
    convert y array to one-hot array
    :param y:[batch size,z,x,y,channel]
    :param numclass:number class
    :return:[batch size,z,x,y,numclass]
    """
    one_hoty = np.reshape(y, (-1,))
    one_hoty = np.eye(numclass)[one_hoty.reshape(-1).astype(np.int)]
    return one_hoty


class Vnet3dModuleMultiLabel(object):
    """
        A Vnet3dMultiLabel implementation
        :param image_height: number of height in the input image
        :param image_width: number of width in the input image
        :param image_depth: number of depth in the input image
        :param channels: number of channels in the input image
        :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_height, image_width, image_depth, channels=4, numclass=4,
                 costname=("categorical_crossentropy",), inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels
        self.numclass = numclass
        self.labelchannels = 1

        self.X = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                   self.numclass])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')

        self.weight_loss = [0.1, 1., 1., 1.]
        self.Y_pred = _create_conv_net(self.X, self.image_depth, self.image_width, self.image_height, self.channels,
                                       self.phase, self.drop, self.numclass)

        self.cost1 = self.__get_cost(self.Y_pred, self.Y_gt, costname[0], gamma=2)
        self.cost_re = multiscalessim2d_loss(self.Y_pred, self.Y_gt, self.numclass - 1)
        self.cost = self.cost1 + self.cost_re
        self.Y_pred_arg = tf.reshape(tf.argmax(self.Y_pred, axis=-1),
                                     (-1, self.image_depth, self.image_height, self.image_width, self.labelchannels))
        self.accuracy = self.__get_metrics(self.Y_pred, self.Y_gt, "mdice")

        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)
        else:
            self.sess = tf.InteractiveSession(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    def __get_cost(self, Y_pred, Y_gt, cost_name, gamma=2):
        if cost_name == "categorical_crossentropy":
            loss = categorical_crossentropy(Y_pred, Y_gt)
        if cost_name == "weighted_categorical_crossentropy":
            loss = weighted_categorical_crossentropy(Y_pred, Y_gt, self.weight_loss)
        if cost_name == "categorical_dice":
            loss = categorical_dice(Y_pred, Y_gt, self.weight_loss)
        if cost_name == "generalized_dice_loss_w":
            loss = generalized_dice_loss_w(Y_pred, Y_gt)
        if cost_name == "categorical_focal_loss":
            loss = categorical_focal_loss(Y_pred, Y_gt, gamma, self.weight_loss)
        if cost_name == "categorical_tversky":
            loss = categorical_tversky(Y_pred, Y_gt, beta=0.25, weight_loss=self.weight_loss)
        if cost_name == "categorical_dicePcrossentroy":
            loss = categorical_dicePcrossentroy(Y_pred, Y_gt, self.weight_loss)
        if cost_name == "categorical_dicePfocalloss":
            loss = categorical_dicePfocalloss(Y_pred, Y_gt, self.weight_loss, 0.6, 3.)
        return loss

    def __get_metrics(self, Y_pred, Y_gt, metric_name="miou"):
        if metric_name == "miou":
            metric = mean_iou(Y_pred, Y_gt)
        if metric_name == "mdice":
            metric = mean_dice(Y_pred, Y_gt)
        return metric

    def __loadnumtraindata(self, train_images, train_lanbels, num_sample, num_sample_index_in_epoch):
        subbatch_xs = np.empty((num_sample, self.image_depth, self.image_height, self.image_width, self.channels))
        subbatch_ys = np.empty((num_sample, self.image_depth, self.image_height, self.image_width, self.labelchannels))
        batch_xs_path, batch_ys_path, num_sample_index_in_epoch = _next_batch(train_images, train_lanbels,
                                                                              num_sample, num_sample_index_in_epoch)
        for num in range(len(batch_xs_path)):
            image = np.load(batch_xs_path[num])
            label = np.load(batch_ys_path[num])
            # prepare 3 model output
            batch_ys_tmp = np.zeros(label.shape, np.int)
            batch_ys_tmp[label == 1.] = 1
            batch_ys_tmp[label == 2.] = 2
            batch_ys_tmp[label == 4.] = 3
            subbatch_xs[num, :, :, :, :] = np.reshape(image,
                                                      (self.image_depth, self.image_height, self.image_width,
                                                       self.channels))
            subbatch_ys[num, :, :, :, :] = np.reshape(batch_ys_tmp,
                                                      (self.image_depth, self.image_height, self.image_width,
                                                       self.labelchannels))
        subbatch_xs = subbatch_xs.astype(np.float)
        subbatch_ys = subbatch_ys.astype(np.float)
        return subbatch_xs, subbatch_ys, num_sample_index_in_epoch

    def train(self, train_images, train_labels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1, showwind=[6, 8]):
        num_sample = 1
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model\\"):
            os.makedirs(logs_path + "model\\")
        model_path = logs_path + "model\\" + model_path
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        self.sess.run(init)

        ckpt = tf.train.get_checkpoint_state(logs_path + "model\\")
        if ckpt and ckpt.model_checkpoint_path:
            print('Checkpoint file: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        DISPLAY_STEP = 1
        num_sample_index_in_epoch = 0
        index_in_epoch = 0
        train_epochs = train_images.shape[0] * train_epochs
        for i in range(train_epochs):
            # Extracting num_sample images and labels from given data
            if i % num_sample == 0 or i == 0:
                subbatch_xs, subbatch_ys, num_sample_index_in_epoch = self.__loadnumtraindata(train_images,
                                                                                              train_labels, num_sample,
                                                                                              num_sample_index_in_epoch)
            # get new batch
            batch_xs, batch_ys, index_in_epoch = _next_batch(subbatch_xs, subbatch_ys, batch_size, index_in_epoch)
            # convert label to one hot type
            batch_ys_onehot = convert_to_one_hot(batch_ys, self.numclass)
            batch_ys_onehot = np.reshape(batch_ys_onehot,
                                         (-1, self.image_depth, self.image_height, self.image_width, self.numclass))
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = self.sess.run([self.cost, self.accuracy],
                                                           feed_dict={self.X: batch_xs,
                                                                      self.Y_gt: batch_ys_onehot,
                                                                      self.lr: learning_rate,
                                                                      self.phase: 1,
                                                                      self.drop: dropout_conv})
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))

                pred_arg = self.sess.run(self.Y_pred_arg, feed_dict={self.X: batch_xs,
                                                                     self.Y_gt: batch_ys_onehot,
                                                                     self.phase: 1,
                                                                     self.drop: 1})
                batch_ys_tmp = np.argmax(batch_ys_onehot, axis=-1)
                gt = np.reshape(batch_ys_tmp[0], (self.image_depth, self.image_height, self.image_width))
                gt = gt.astype(np.float)
                save_images(gt, showwind, path=logs_path + 'gt_%d_epoch.png' % (i), pixelvalue=85)

                result = np.reshape(pred_arg[0], (self.image_depth, self.image_height, self.image_width))
                result = result.astype(np.float)
                save_images(result, showwind, path=logs_path + 'predict_%d_epoch.png' % (i), pixelvalue=85)

                save_path = saver.save(self.sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = self.sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                                 self.Y_gt: batch_ys_onehot,
                                                                                 self.lr: learning_rate,
                                                                                 self.phase: 1,
                                                                                 self.drop: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(self.sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        test_images = np.reshape(test_images,
                                 (test_images.shape[0], test_images.shape[1], test_images.shape[2], self.channels))
        test_images = test_images.astype(np.float)
        y_dummy = np.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], self.numclass))
        pred_arg = self.sess.run(self.Y_pred_arg, feed_dict={self.X: [test_images],
                                                             self.Y_gt: [y_dummy],
                                                             self.phase: 1,
                                                             self.drop: 1})
        result = pred_arg.astype(np.float)
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (test_images.shape[0], test_images.shape[1], test_images.shape[2]))
        return result
