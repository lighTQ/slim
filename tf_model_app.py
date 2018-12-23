# coding:utf-8

import sys, os
import tensorflow as tf
import numpy as np
import cv2
import argparse
from  image_classifier.diyTrain_v2 import  build_model
import  matplotlib.pyplot as plt

def parse_arguments(argv):
    ap = argparse.ArgumentParser()

    ap.add_argument('-testDataset', type=str, nargs='+', help='use offered test file path')
    ap.add_argument('-width', type=int, nargs='+', help='img_resize_width')
    ap.add_argument('-height', type=int, nargs='+', help='img__resize_height')
    ap.add_argument('-channel_num', type=int, nargs='+', help='channels_num')
    ap.add_argument('-model_path', type=str, nargs='+',
                    help='in order to load the file ,which is the abs path')
    ap.add_argument('-typefile', type=str, nargs='+', help='label name path ')
    return vars(ap.parse_args())


def preprocessImageFolder(imagePath, width=128, height=128):
    print("processing image....")
    test_imgFiles = []
    X = []
    if os.path.isdir(imagePath):
        print(os.path.isdir(imagePath))
        print("测试图片文件夹。。。。")
        for imgfile in os.listdir(imagePath):
            img = os.path.join(imagePath, imgfile)
            if (img.lower().endswith('jpg') or img.lower().endswith('jpeg') or img.lower().endswith('png')):
                test_imgFiles.append(img)  # JAVA调用的path
                # print(img)
                img = cv2.resize(cv2.imread(img), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                # test_imgFiles.append(img)  #  plt 打印使用的ndarray
                X.append(img)
            else:
                tf.logging.error("file type is not support")
                sys.exit("program exited!")
        X = np.array(X).astype('float32') / 255
        print(X.shape)
        return X, test_imgFiles

    elif (os.path.isfile(imagePath)):
        print(os.path.isfile(imagePath))
        img = imagePath.lower()
        print(img)
        if (img.endswith('jpg') or img.endswith('jpeg') or img.endswith('png')):
            print("测试单张图片")
            test_imgFiles.append(imagePath)  # plt show 时加上注释
            img = cv2.resize(cv2.imread(imagePath), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X = np.array(img).astype('float32') / 255
            # test_imgFiles.append(img)  # plt show使用时去掉注释
            X = X.reshape((1,) + X.shape)
            print(X.shape)
            return X, test_imgFiles
        else:
            tf.logging.error(" input image file  type is not support")
            sys.exit(" program exited!")
    else:
        tf.logging.error(" input must be a directory or a image File  ")
        sys.exit("program exited !")


def restore_model(testPicArr, width, height, channel=3):
    with tf.Graph().as_default() as tg:
        datas_placeholder, labels_placeholder, dropout_placeholdr, y = build_model(width, height, channel)

        logits = tf.nn.softmax(y)
        preValue = tf.argmax(logits, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, model_file)
            net, logits, preValue = sess.run([y, logits, preValue], feed_dict={datas_placeholder: testPicArr})
            np.set_printoptions(precision=2, suppress=True)
            # print('全连接层输出：  _________>>>>>>>>>>>________>\r\n',net)
            # print('softmax 层输出： _________>>>>>>>>>>>________>\r\n',logits)
            return preValue


if __name__ == '__main__':


    #******* script parameters   **********#

    '''
-testDataset
E:/AI_Codes/tensorflow_Task/image_classifier/101Categories/ant
-width
128
-height
128
-channel_num
3
-model_path
E:/AI_Codes/tensorflow_Task/image_classifier/model
-typefile
E:/AI_Codes/tensorflow_Task/image_classifier/101Categories
    
    
    '''
    # step1 ： 获取动态参数
    arguments = parse_arguments(sys.argv[1:])
    width = arguments['width'][0]
    height = arguments['height'][0]
    model_path = arguments['model_path'][0]
    typefile = arguments['typefile'][0]
    testDataset = arguments['testDataset'][0]


    # step2  加载数据
    X, ImgFiles = preprocessImageFolder(testDataset, width, height)

    # step3 : 获取预测的类别标签名称：
    types = []
    tf.logging.info("正在获取数据标签.....")
    for ls in os.listdir(typefile):
        print(ls)
        types.append(ls)
    # print(types)

    # step 4 生成返回的字典
    back_testResult = {}
    #  step 5 调用模型进行预测
    classes = restore_model(X, width, height)
    print('预测类别： ', classes)
    print("predicted result in your kerboard floder is： \n")
    for i, index in enumerate(classes):
        key = str(ImgFiles[i])
        value = str(types[index])
        back_testResult.setdefault(key, value)
        # plt.imshow(ImgFiles[i])
        # plt.title(str(types[index]))
        # plt.show()
    print(back_testResult)
