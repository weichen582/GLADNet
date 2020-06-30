from __future__ import print_function

import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *

def FG(input_im):
  with tf.compat.v1.variable_scope('FG'):
    input_rs = tf.image.resize(input_im, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    p_conv1 = tf.compat.v1.layers.conv2d(input_rs, 64, 3, 2, padding='same', activation=tf.nn.relu) # 48
    p_conv2 = tf.compat.v1.layers.conv2d(p_conv1,  64, 3, 2, padding='same', activation=tf.nn.relu) # 24
    p_conv3 = tf.compat.v1.layers.conv2d(p_conv2,  64, 3, 2, padding='same', activation=tf.nn.relu) # 12
    p_conv4 = tf.compat.v1.layers.conv2d(p_conv3,  64, 3, 2, padding='same', activation=tf.nn.relu) # 6
    p_conv5 = tf.compat.v1.layers.conv2d(p_conv4,  64, 3, 2, padding='same', activation=tf.nn.relu) # 3
    p_conv6 = tf.compat.v1.layers.conv2d(p_conv5,  64, 3, 2, padding='same', activation=tf.nn.relu) # 1

    p_deconv1 = tf.image.resize(p_conv6, (3, 3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv1 = tf.compat.v1.layers.conv2d(p_deconv1, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv1 = p_deconv1 + p_conv5
    p_deconv2 = tf.image.resize(p_deconv1, (6, 6), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv2 = tf.compat.v1.layers.conv2d(p_deconv2, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv2 = p_deconv2 + p_conv4
    p_deconv3 = tf.image.resize(p_deconv2, (12, 12), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv3 = tf.compat.v1.layers.conv2d(p_deconv3, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv3 = p_deconv3 + p_conv3
    p_deconv4 = tf.image.resize(p_deconv3, (24, 24), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv4 = tf.compat.v1.layers.conv2d(p_deconv4, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv4 = p_deconv4 + p_conv2
    p_deconv5 = tf.image.resize(p_deconv4, (48, 48), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv5 = tf.compat.v1.layers.conv2d(p_deconv5, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv5 = p_deconv5 + p_conv1
    p_deconv6 = tf.image.resize(p_deconv5, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    p_deconv6 = tf.compat.v1.layers.conv2d(p_deconv6, 64, 3, 1, padding='same', activation=tf.nn.relu)

    p_output = tf.image.resize(p_deconv6, (tf.shape(input=input_im)[1], tf.shape(input=input_im)[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    a_input = tf.concat([p_output, input_im], axis=3)
    a_conv1 = tf.compat.v1.layers.conv2d(a_input, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv2 = tf.compat.v1.layers.conv2d(a_conv1, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv3 = tf.compat.v1.layers.conv2d(a_conv2, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv4 = tf.compat.v1.layers.conv2d(a_conv3, 128, 3, 1, padding='same', activation=tf.nn.relu)
    a_conv5 = tf.compat.v1.layers.conv2d(a_conv4, 3,   3, 1, padding='same', activation=tf.nn.relu)
    return a_conv5



class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.base_lr = 0.001

        self.input_low = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        self.output = FG(self.input_low)
        self.loss = tf.reduce_mean(input_tensor=tf.abs((self.output - self.input_high) * [[[[0.11448, 0.58661, 0.29891]]]]))

        self.global_step = tf.Variable(0, trainable = False)
        self.lr = tf.compat.v1.train.exponential_decay(self.base_lr, self.global_step, 100, 0.96)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        print("[*] Initialize model successfully...")

    def evaluate(self, epoch_num, eval_low_data, sample_dir):
        print("[*] Evaluating for epoch %d..." % (epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            result = self.sess.run(self.output, feed_dict={self.input_low: input_low_eval})
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx + 1, epoch_num)), input_low_eval, result)


    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, sample_dir, ckpt_dir, eval_every_epoch):

        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        load_model_status, global_step = self.load(self.saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir)
                self.save(self.saver, iter_num, ckpt_dir, "GLADNet")

        print("[*] Finish training")

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir):
        tf.compat.v1.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status, _ = self.load(self.saver, './model/')
        if load_model_status:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            start_time = time.time()
            result = self.sess.run(self.output, feed_dict = {self.input_low: input_low_test})
            total_run_time += time.time() - start_time
            save_images(os.path.join(save_dir, name + "_glad."   + suffix), result)

        ave_run_time = total_run_time / float(len(test_low_data))
        print("[*] Average run time: %.4f" % ave_run_time)
