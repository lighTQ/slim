# coding:utf-8

import os
import cv2
from keras.utils import np_utils
# 图像读取库
from PIL import Image
# 矩阵运算库
import numpy as np
import tensorflow as tf
from  sklearn.model_selection import train_test_split
from  tensorflow.contrib import slim
from  sklearn.metrics import  *


# from slim_opt.slimApp import slim_model
from nets import nets_factory
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory




# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1

def get_files(img_src, width, height, validate_percent, test_percent):
    X, y = [], []
    categorical_num = len(set(os.listdir(img_src)))
    for i, folder in enumerate(os.listdir(img_src)):
        print(i, folder)
        for index, filename in enumerate(os.listdir(os.path.join(img_src, folder))):
            # imgFile = os.path.join(img_src, (folder+"/"+filename))
            imgFile = os.path.join(img_src, folder, filename)
            print(index, imgFile)
            # tf.image.resize_image_with_crop_or_pad(imgFile, width, height)
            img = cv2.resize(cv2.imread(imgFile), dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            X.append(img)
            # X.append(imgFile)
            y.append(i)
    X = np.array(X).astype('float32') / 255
    # y = np.array(y)

    y = np_utils.to_categorical(y, categorical_num)
    # y = np_utils.to_categorical(y).reshape(-1, categorical_num)

    print(X.shape)
    print(y.shape)

    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validate_percent, shuffle=True)
    print('preprocessing images is OK..')

    return categorical_num, X_train, X_valid, X_test, y_train, y_valid, y_test


def next_batch_set(images, labels, batch_size=16):
    """Generate a batch training data.

    Args:
        images: A 4-D array representing the training images.
        labels: A 1-D array representing the classes of images.
        batch_size: An integer.

    Return:
        batch_images: A batch of images.
        batch_labels: A batch of labels.
    """
    indices = np.random.choice(len(images), batch_size)
    batch_images = images[indices]
    batch_labels = labels[indices]
    return batch_images, batch_labels


# 计算有多少类图片
def build_model(width, height, categorical_num):

    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, width, height, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None, categorical_num])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)

    net = slim.conv2d(datas_placeholder, 20, [5, 5], activation_fn=tf.nn.relu)
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.conv2d(net, 40, [4, 4], activation_fn=tf.nn.relu)
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu)
    net = slim.dropout(net, 0.5)
    logits = slim.fully_connected(net, categorical_num, activation_fn=None)

    return datas_placeholder, labels_placeholder, dropout_placeholdr, logits

# 配置优化器：

def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


# 模型评估指标类别
def metrics_result(y_label, y_pred):
    _recall = recall_score(y_label, y_pred, average='weighted')
    _precision = precision_score(y_label, y_pred, average='weighted')
    _f1 = f1_score(y_label, y_pred, average='macro')
    _acc = accuracy_score(y_label, y_pred)
    _cm = confusion_matrix(y_label, y_pred)
    return _recall, _precision, _f1, _acc, _cm



# 返回评估指标
def call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test,logits):


    train_feed_dict = {
                datas_placeholder: X_train,
                labels_placeholder: y_train,
                dropout_placeholdr: 0
            }
    valid_feed_dict = {
        datas_placeholder: X_valid,
        labels_placeholder: y_valid,
        dropout_placeholdr: 0
    }
    test_feed_dict = {
        datas_placeholder: X_test,
        labels_placeholder: y_test,
        dropout_placeholdr: 0
    }

    predicted_labels = tf.argmax(logits, axis=1)

    # metrics show
    y_train = np.argmax(y_train,axis=1)
    y_valid = np.argmax(y_valid,axis=1)
    y_test = np.argmax(y_test,axis=1)


    # 测试集预测值
    predicted_labels_train = sess.run(predicted_labels, feed_dict=train_feed_dict)
    predicted_labels_valid = sess.run(predicted_labels, feed_dict=valid_feed_dict)
    predicted_labels_test = sess.run(predicted_labels, feed_dict=test_feed_dict)

    # print('teye of y, shape of y', type(y_test), y_test.shape)
    # print('teye of y_pred, shape of y_pred', type(predicted_labels_val), predicted_labels_val.shape)

    # test_correct_prediction = tf.equal(predicted_labels_test, y_test)
    # # 准确率
    # test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    # print('测试集准确率：', sess.run(test_accuracy))



    metric_train = metrics_result(y_train, predicted_labels_train)
    metric_valid = metrics_result(y_valid, predicted_labels_valid)
    metric_test = metrics_result(y_test, predicted_labels_test)

    # print(type(metric_test), metric_test)

    #
    # 调用metric_result 方法返回的是tuple： 元素顺序为：  _recall, _precision, _f1, _acc, _cm
    call_res = {
        # 'model_id': modelId, 'model_userid': model_userid, 'model_version': model_version,
        #         'ams_id': ams_id,
        #         'calculationType': 'result',
                'train_recall_score': metric_train[0], 'train_precision_score': metric_train[1],
                'train_f1_score': metric_train[2], 'train_acc': metric_train[3],
                'train_cm': metric_train[4].tolist(),
                'valid_recall_score': metric_valid[0], 'valid_precision_score': metric_valid[1],
                'valid_f1_score': metric_valid[2], 'valid_acc': metric_valid[3],
                'valid_cm': metric_valid[4].tolist(),
                'test_recall_score': metric_test[0], 'test_precision_score': metric_test[1],
                'test_f1_score': metric_test[2], 'test_acc': metric_test[3], 'test_cm': metric_test[4].tolist()}

    print(call_res)
    return  call_res




if __name__ == '__main__':

    tf.app.flags.DEFINE_string(
        'model_name', 'vgg_16', 'The name of the architecture to train.')

    tf.app.flags.DEFINE_integer(
        'train_image_size', 224, 'Train image size')

    ######################
    # Optimization Flags #
    ######################

    tf.app.flags.DEFINE_float(
        'weight_decay', 0.00004, 'The weight decay on the model weights.')

    tf.app.flags.DEFINE_string(
        'optimizer', 'rmsprop',
        'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
        '"ftrl", "momentum", "sgd" or "rmsprop".')

    tf.app.flags.DEFINE_float(
        'adadelta_rho', 0.95,
        'The decay rate for adadelta.')

    tf.app.flags.DEFINE_float(
        'adagrad_initial_accumulator_value', 0.1,
        'Starting value for the AdaGrad accumulators.')

    tf.app.flags.DEFINE_float(
        'adam_beta1', 0.9,
        'The exponential decay rate for the 1st moment estimates.')

    tf.app.flags.DEFINE_float(
        'adam_beta2', 0.999,
        'The exponential decay rate for the 2nd moment estimates.')

    tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

    tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                              'The learning rate power.')

    tf.app.flags.DEFINE_float(
        'ftrl_initial_accumulator_value', 0.1,
        'Starting value for the FTRL accumulators.')

    tf.app.flags.DEFINE_float(
        'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

    tf.app.flags.DEFINE_float(
        'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

    tf.app.flags.DEFINE_float(
        'momentum', 0.9,
        'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

    tf.app.flags.DEFINE_integer(
        'quantize_delay', -1,
        'Number of steps to start quantized training. Set to -1 would disable '
        'quantized training.')

    #######################
    # Learning Rate Flags #
    #######################

    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

    tf.app.flags.DEFINE_float(
        'end_learning_rate', 0.0001,
        'The minimal end learning rate used by a polynomial decay learning rate.')

    tf.app.flags.DEFINE_float(
        'label_smoothing', 0.0, 'The amount of label smoothing.')

    tf.app.flags.DEFINE_float(
        'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

    tf.app.flags.DEFINE_float(
        'num_epochs_per_decay', 2.0,
        'Number of epochs after which learning rate decays. Note: this flag counts '
        'epochs per clone but aggregates per sync replicas. So 1.0 means that '
        'each clone will go over full epoch individually, but replicas will go '
        'once across all replicas.')

    tf.app.flags.DEFINE_bool(
        'sync_replicas', False,
        'Whether or not to synchronize the replicas during training.')

    tf.app.flags.DEFINE_integer(
        'replicas_to_aggregate', 1,
        'The Number of gradients to collect before updating params.')

    tf.app.flags.DEFINE_float(
        'moving_average_decay', None,
        'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.')

    FLAGS = tf.app.flags.FLAGS

    width, height = FLAGS.train_image_size, FLAGS.train_image_size

    # 数据文件夹
    data_dir = "D:\\AI_codes\\models-master(1)\\models-master\\research\\slim\\101Category"
    # data_dir="E:\\AI_test\\tensorflow_Task\\image_classifier\\training_data"

    num_classes, X_train, X_valid, X_test, y_train, y_valid, y_test = get_files(data_dir, width, height, 0.1, 0.2)

    # 训练还是测试
    train = True
    # 模型文件路径
    model_path = "E:\\AI_test\\tensorflow_Task\\image_classifier\\model_inception_v4"
    # model_path = "E:\AI_test\\tensorflow_Task\\image_classifier\\model_training_data"

    model_name = "v1.0"

    # datas_placeholder, labels_placeholder, dropout_placeholdr, logits = slim_model(width,height,num_classes)
    # datas_placeholder, labels_placeholder, dropout_placeholdr, logits = build_model(width,height,num_classes)

    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, width, height, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None, num_classes])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)


    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    train_image_size = FLAGS.train_image_size or network_fn.default_image_size


    logits, end_points = network_fn(datas_placeholder)

    global_step = tf.Variable(0)

    # 平均损失
    mean_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_placeholder,  # 真实值
        logits=logits  # 预测值
    ))
    # mean_loss = tf.reduce_mean(slim.losses.softmax_cross_entropy(logits,labels_placeholder))

    # 自定义优化器
    optimizer = _configure_optimizer(FLAGS.learning_rate)
    print(optimizer.get_name())
    train_op = optimizer.minimize(mean_loss, global_step=global_step)

    # 定义优化器，指定要优化的损失函数
    # train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(mean_loss,global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(labels_placeholder, 1))

    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 用于保存和载入模型
    saver = tf.train.Saver(max_to_keep=3)
    max_acc = 0
    with tf.Session() as sess:

        if train:
            print("训练模式")
            # 如果是训练，初始化参数
            sess.run(tf.global_variables_initializer())

            #实现断点续训
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            train_images, train_labels = next_batch_set(X_train, y_train, 8)
            valid_images, valid_labels = next_batch_set(X_valid, y_valid, 8)
            print('batch size of train dataset :', train_labels.shape, train_images.shape)


            # 定义输入和Label以填充容器，训练时dropout为0.25
            train_feed_dict = {
                datas_placeholder: train_images,
                labels_placeholder: train_labels,
                dropout_placeholdr: 0.1
            }

            valid_feed_dict = {
                datas_placeholder: valid_images,
                labels_placeholder: valid_labels,
                dropout_placeholdr: 0.0
            }

            print("训练集数据大小： ")
            print(X_train.shape, y_train.shape)
            #
            top3_acc_list =[]
            for i in range(180):

                # 训练模型：
                sess.run(train_op, feed_dict=train_feed_dict)

                mean_loss_train, train_accuracy,step = sess.run([ mean_loss, accuracy,global_step], feed_dict=train_feed_dict)
                mean_loss_val, val_accuracy,step = sess.run([ mean_loss, accuracy,global_step], feed_dict=valid_feed_dict)

                top3_acc_list.append(str(i+1)+'、val_acc: '+str(val_accuracy)+'\n')
                np.set_printoptions(precision=3,suppress=True)
                if i % 10 == 0:
                    print("step = {}\ttrain_mean loss = {}\t train_acc = {}\tval_mean loss = {}\t val_acc = {} ".format(step, mean_loss_train, train_accuracy, mean_loss_val, val_accuracy))

                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                # 保存验证集上精度最高的三个模型
                if val_accuracy > max_acc:
                    max_acc = val_accuracy
                    saver.save(sess, os.path.join(model_path,model_name),global_step=global_step)
            print("训练结束，保存模型到{}".format(model_path))
            print('top3 acc list :', top3_acc_list)

        else:
            print("测试模式")

            print("测试集数据大小： ")
            print(X_test.shape, y_test.shape)

            model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(sess,model_file)

            # way 1
            # ckpt = tf.train.get_checkpoint_state(model_path)
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)


            # # 如果是测试，载入参数
            # saver.restore(sess, model_path)
            print("从{}载入模型".format(model_path))

            # 返回模型评估指标：
            call_back_metrics(X_train, X_valid, X_test, y_train, y_valid, y_test,logits)

            # label和名称的对照关系
            label_name_dict = {
                # 0: "手风琴",
                # 1: "椅子",
                # 2: "蚂蚁"
            }

            # 构造类别标签
            for index, name in enumerate(os.listdir(data_dir)):
                label_name_dict.setdefault(index, name)



            # acc_score = accuracy_score(y_test, predicted_labels_val)
            # recall_score = recall_score(y_test, predicted_labels_val,average='weighted')
            # precision_score = precision_score(y_test, predicted_labels_val,average='weighted')
            # f1_score = f1_score(y_test, predicted_labels_val,average='weighted')
            #
            # # _acc_score, _recall_score, _precision_score, _f1_score = sess.run(
            # #     [acc_score, recall_score, precision_score, f1_score],feed_dict=test_feed_dict)
            #
            # print(' 测试集模型评估指标：\r\n acc score:{}\t recall score:{}\t precision score:{}\t f1 score:{}'.format(acc_score, recall_score,
            #                                                                                  precision_score,
            #                                                                                  f1_score))


            # print(')))---------show result ----------(((')
            # print("-------[[[[[[]]]]]--------\n"*7)
            #
            # # 真实label与模型预测label
            # for real_label, predicted_label in zip(y_test, predicted_labels_val):
            #     # 将label id转换为label名
            #     real_label_name = label_name_dict[real_label]
            #     print('real_label : ', real_label, '  real_label_name: ', real_label_name)
            #     predicted_label_name = label_name_dict[predicted_label]
            #     print('predict label: ', predicted_label, ' predict_label_name: ', predicted_label_name, "\n")
            #
            #     print("{} => {}".format(real_label_name, predicted_label_name))


