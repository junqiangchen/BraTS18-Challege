'''

'''
from Vnet.layer import (conv_bn_relu_drop, down_sampling, deconv_relu, crop_and_concat, resnet_Add, conv_sigmod,
                        save_images)
import tensorflow as tf
import numpy as np
import os


def _create_conv_net(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 128, 256), scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 256, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 64, 128), scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat(layer3, deconv2)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 128, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 32, 64), scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 64, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_3')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 16, 32), scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 16, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 16, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_3')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigmod(x=layer9, kernal=(1, 1, 1, 16, n_class), scope='output')
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


class Vnet3dModule(object):
    """
        A VNet3d implementation
        :param image_height: number of height in the input image
        :param image_width: number of width in the input image
        :param image_depth: number of depth in the input image
        :param channels: number of channels in the input image
        :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_height, image_width, image_depth, channels=1, numclass=1, costname=("dice coefficient",),
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels
        self.numclass = numclass

        self.X = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                   self.numclass])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')

        self.Y_pred = _create_conv_net(self.X, self.image_depth, self.image_width, self.image_height, self.channels,
                                       self.phase, self.drop, self.numclass)
        self.cost = self.__get_cost(self.Y_pred, self.Y_gt, costname[0])
        self.accuracy = -self.cost

        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_cost(self, Y_pred, Y_gt, cost_name):
        Z, H, W, C = Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
            true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
        return loss

    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1, showwindow=[8, 8]):
        num_sample = 100
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
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        if os.path.exists(model_path):
            saver.restore(sess, model_path)

        # load data and show result param
        DISPLAY_STEP = 1
        num_sample_index_in_epoch = 0
        index_in_epoch = 0

        train_epochs = train_images.shape[0] * train_epochs

        subbatch_xs = np.empty((num_sample, self.image_depth, self.image_height, self.image_width, self.channels))
        subbatch_ys = np.empty((num_sample, self.image_depth, self.image_height, self.image_width, self.numclass))

        for i in range(train_epochs):
            # Extracting num_sample images and labels from given data
            if i % num_sample == 0 or i == 0:
                batch_xs_path, batch_ys_path, num_sample_index_in_epoch = _next_batch(train_images, train_lanbels,
                                                                                      num_sample,
                                                                                      num_sample_index_in_epoch)
                for num in range(len(batch_xs_path)):
                    image = np.load(batch_xs_path[num])
                    label = np.load(batch_ys_path[num])
                    # prepare 3 model output
                    batch_ys1 = label.copy()
                    batch_ys1[label == 1.] = 1.
                    batch_ys1[label != 1.] = 0.
                    batch_ys2 = label.copy()
                    batch_ys2[label == 2.] = 1.
                    batch_ys2[label != 2.] = 0.
                    batch_ys3 = label.copy()
                    batch_ys3[label == 4.] = 1.
                    batch_ys3[label != 4.] = 0.
                    subbatch_xs[num, :, :, :, :] = np.reshape(image,
                                                              (self.image_depth, self.image_height, self.image_width,
                                                               self.channels))
                    label_ys = np.empty((self.image_depth, self.image_height, self.image_width, self.numclass))
                    label_ys[:, :, :, 0] = batch_ys1
                    label_ys[:, :, :, 1] = batch_ys2
                    label_ys[:, :, :, 2] = batch_ys3
                    subbatch_ys[num, :, :, :, :] = np.reshape(label_ys,
                                                              (self.image_depth, self.image_height, self.image_width,
                                                               self.numclass))

                subbatch_xs = subbatch_xs.astype(np.float)
                subbatch_ys = subbatch_ys.astype(np.float)
            # get new batch
            batch_xs, batch_ys, index_in_epoch = _next_batch(subbatch_xs, subbatch_ys, batch_size, index_in_epoch)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run(
                    [self.cost, self.accuracy], feed_dict={self.X: batch_xs,
                                                           self.Y_gt: batch_ys,
                                                           self.lr: learning_rate,
                                                           self.phase: 1,
                                                           self.drop: dropout_conv})
                print('epochs %d training_loss ,training_accuracy ''=> %.5f,%.5f ' % (i, train_loss, train_accuracy))

                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                        self.Y_gt: batch_ys,
                                                        self.phase: 1,
                                                        self.drop: 1})
                gt = np.reshape(batch_ys[0], (self.image_depth, self.image_height, self.image_width, self.numclass))
                gt1 = gt[:, :, :, 0]
                gt1 = np.reshape(gt1, (self.image_depth, self.image_height, self.image_width))
                gt1 = gt1.astype(np.float)
                save_images(gt1, showwindow, path=logs_path + 'gt1_%d_epoch.png' % i)
                gt2 = gt[:, :, :, 1]
                gt2 = np.reshape(gt2, (self.image_depth, self.image_height, self.image_width))
                gt2 = gt2.astype(np.float)
                save_images(gt2, showwindow, path=logs_path + 'gt2_%d_epoch.png' % i)
                gt3 = gt[:, :, :, 2]
                gt3 = np.reshape(gt3, (self.image_depth, self.image_height, self.image_width))
                gt3 = gt3.astype(np.float)
                save_images(gt3, showwindow, path=logs_path + 'gt3_%d_epoch.png' % i)

                result = np.reshape(pred[0], (self.image_depth, self.image_height, self.image_width, self.numclass))
                result1 = result[:, :, :, 0]
                result1 = np.reshape(result1, (self.image_depth, self.image_height, self.image_width))
                result1 = result1.astype(np.float)
                save_images(result1, showwindow, path=logs_path + 'predict1_%d_epoch.png' % i)
                result2 = result[:, :, :, 1]
                result2 = np.reshape(result2, (self.image_depth, self.image_height, self.image_width))
                result2 = result2.astype(np.float)
                save_images(result2, showwindow, path=logs_path + 'predict2_%d_epoch.png' % i)
                result3 = result[:, :, :, 2]
                result3 = np.reshape(result3, (self.image_depth, self.image_height, self.image_width))
                result3 = result3.astype(np.float)
                save_images(result3, showwindow, path=logs_path + 'predict3_%d_epoch.png' % i)

                save_path = saver.save(sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        test_images = np.reshape(test_images,
                                 (test_images.shape[0], test_images.shape[1], test_images.shape[2], self.channels))
        test_images = test_images.astype(np.float)
        y_dummy = np.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 3))
        pred = self.sess.run(self.Y_pred, feed_dict={self.X: [test_images], self.Y_gt: [y_dummy], self.phase: 1,
                                                     self.drop: 1})
        result = pred.astype(np.float) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        result = np.reshape(result, (test_images.shape[0], test_images.shape[1], test_images.shape[2], self.numclass))
        return result
