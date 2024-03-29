from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import cv2
from PIL import Image
from get_segment import check_full_body, check_background_color, check_lookbook

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from LIP_model import *

N_CLASSES = 20
INPUT_SIZE = (384, 384)
NUM_STEPS = 10000  # Number of images in the validation set.

DATA_DIRECTORY = '/Users/macuser/Downloads/dataset/LOOKBOOK/lookbook/data/'
DATA_LIST_PATH = '/Users/macuser/PycharmProjects/Processing_LOOKBOOK/duongpd/Todo_list_Tien2.txt'
RESTORE_DIR = '/Users/macuser/Downloads/JPPNet-s2/model.ckpt-205632'
OUTPUT_DIR = '/Users/macuser/Desktop'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list
        NUM_STEPS = len(image_list)
    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100,
                                                            name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100,
                                                            name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075,
                                                            name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075,
                                                            name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125,
                                                            name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125,
                                                            name='fc3_parsing')

    parsing_out1 = tf.reduce_mean(
        tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)
    parsing_out2 = tf.reduce_mean(
        tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)
    parsing_out3 = tf.reduce_mean(
        tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3, ]),
                  tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3, ])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)

    loader.restore(sess, RESTORE_DIR)
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        start = time.time()
        parsing_ = sess.run(pred_all)

        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        # full body
        if (check_full_body(parsing_im)):
            print("Step {} takes {} second(s) --- JPP -> Saved.".format(step, time.time() - start))
            with open(os.path.join(OUTPUT_DIR, 'filter.txt'), 'a') as writer:
                print(reader.tmp[step])
                writer.write("{}\n".format(reader.tmp[step]))
        with open(os.path.join(OUTPUT_DIR, 'check_flag.txt'), 'a') as writer:
            writer.write("{}\n".format(step))

        # if check_background_color(image_list[step], parsing_im): # background as VITON
        #     if check_full_body(parsing_im): # full body
        #         output_dir_full = '/'.join(str(os.path.join(OUTPUT_DIR, 'full_segment', reader.tmp[step])).split('/')[:-1]).replace('img/',
        #                                                                                                       'seg/')
        #         if not os.path.exists(output_dir_full):
        #             os.makedirs(output_dir_full)
        #
        #         parsing_im.save(os.path.join(OUTPUT_DIR, 'full_segment', reader.tmp[step]).replace('img/', 'seg/').replace('jpg', 'png'))
        #         print("Step {} takes {} second(s) --- full_body -> Saved.".format(step, time.time() - start))
        #         with open(os.path.join(OUTPUT_DIR, 'full_segment', 'fullbody_info.txt'), 'a') as writer:
        #             writer.write("{},{}\n".format(reader.tmp[step], reader.tmp[step].replace('img/', 'seg/')))
        #     else: # a half of body
        #         output_dir_half = '/'.join(str(os.path.join(OUTPUT_DIR, 'half_segment',reader.tmp[step])).split('/')[:-1]).replace('img/',
        #                                                                                                  'seg/')
        #         if not os.path.exists(output_dir_half):
        #             os.makedirs(output_dir_half)
        #
        #         parsing_im.save(os.path.join(OUTPUT_DIR, 'half_segment', reader.tmp[step]).replace('img/', 'seg/').replace('jpg', 'png'))
        #         print("Step {} takes {} second(s) --- half_body -> Saved.".format(step, time.time() - start))
        #         with open(os.path.join(OUTPUT_DIR, 'half_segment', 'halfbody_info.txt'), 'a') as writer:
        #             writer.write("{},{}\n".format(reader.tmp[step], reader.tmp[step].replace('img/', 'seg/')))
        #
        # with open(os.path.join(OUTPUT_DIR, 'check_flag.txt'), 'a') as writer:
        #     writer.write("{}\n".format(step))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()