import tensorflow as tf
import numpy as np
import os
import math
from network_res import Model_res
from keras.datasets import cifar10, cifar100
import argparse
import time
import foolbox
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json
from tqdm import tqdm

import pdb

np.random.seed(12345)
tf.set_random_seed(12345)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def cifar(dataset='cifar10'):
    if dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    print('X_train shape:', X_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255
    return X_train, X_test

def attack(threshold):
    resnet101 = models.resnet101(pretrained=True).eval()
    if torch.cuda.is_available():
        resnet101 = resnet101.cuda()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    fmodel = foolbox.models.PyTorchModel(resnet101, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
    labels=['65', '970', '230', '809', '516', '57', '334', '415', '674', '332']
    attack = foolbox.attacks.LinfinityBasicIterativeAttack(fmodel, distance=foolbox.distances.Linfinity, threshold=threshold)

    succ_att = []
    raw_data_path = './raw_data'
    transformer = transforms.Compose([transforms.ToTensor()])

    if os.path.exists('./attack.json'):
        # read attack images
        with open('./attack.json', 'r') as f:
            succ_att = json.loads(f.read())
        print('Load attack images from file.')
    else:
        # generate attack images & save them to file
        with tqdm(total=len(os.listdir(raw_data_path))) as pbar:
            for file_name in os.listdir(raw_data_path):
                ind = int(file_name.split('_')[-1].split('.')[0]) - 1
                true_label = int(labels[ind])

                raw_img_path = os.path.join(raw_data_path, file_name)
                raw_img = cv2.imread(raw_img_path) # (None, None, 3), int
                img = cv2.resize(raw_img, (224, 224)) # (224, 224, 3), int
                img = transformer(img).numpy() # (3, 224, 224), float
                pre_label= np.argmax(fmodel.predictions(img))

                if true_label != pre_label:
                    continue

                adv_image = attack(img, true_label) # (3, 224, 224), float
                if adv_image is None:
                    continue

                diff = adv_image - img
                diff = np.clip(diff, -threshold, threshold) # Linfinity
                adv_image = img + diff # (3, 224, 224), float
                adv_label = np.argmax(fmodel.predictions(adv_image))
                adv_image = np.transpose(adv_image, (1, 2, 0)) # (224, 224, 3), float

                if true_label == adv_label:
                    continue

                succ_att.append([file_name, int(true_label), int(adv_label), adv_image])
                # print('%s, label: %d, predicted class: %d, adversarial class: %d' % tuple(succ_att[-1][:4]))
                pbar.update(1)
        print('New generate attack images, num of attack images: {}'.format(len(succ_att)))

        with open('./attack.json', 'w') as f:
            f.write(json.dumps(succ_att, cls=NumpyEncoder))

    return succ_att, fmodel, transformer

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default=0, type=int, choices=list(range(10)))
    # parser.add_argument('--model', default='res', type=str)
    parser.add_argument('--train_dataset', default='cifar10', type=str)
    parser.add_argument('--n_epoch', default=3, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--test_batch_size', default=10, type=int)
    parser.add_argument('--lambd', default=0.0001, type=float)
    parser.add_argument('--noise_dev', default=20.0, type=float)
    parser.add_argument('--Linfinity', default=0.03, type=float)
    parser.add_argument('--binary_threshold', default=0.5, type=float)
    parser.add_argument('--lr_mode', default=0, type=int)
    parser.add_argument('--test_interval', default=1000, type=int)
    # parser.add_argument('--save_model', default='res_cifar10', type=str)
    args = parser.parse_args()

    if args.test_mode == 0:
        model = Model_res(com_disable=True,rec_disable=True)
        args.save_model = 'normal'
    elif args.test_mode == 1:
        model = Model_res(n_com=1,n_rec=3,com_disable=False,rec_disable=True)
        args.save_model = '1_on_off'
    elif args.test_mode == 2:
        model = Model_res(n_com=2,n_rec=3,com_disable=False,rec_disable=True)
        args.save_model = '2_on_off'
    elif args.test_mode == 3:
        model = Model_res(n_com=3,n_rec=3,com_disable=False,rec_disable=True)
        args.save_model = '3_on_off'
    elif args.test_mode == 4:
        model = Model_res(n_com=3,n_rec=1,com_disable=True,rec_disable=False)
        args.save_model = 'off_on_1'
    elif args.test_mode == 5:
        model = Model_res(n_com=3,n_rec=2,com_disable=True,rec_disable=False)
        args.save_model = 'off_on_2'
    elif args.test_mode == 6:
        model = Model_res(n_com=3,n_rec=3,com_disable=True,rec_disable=False)
        args.save_model = 'off_on_3'
    elif args.test_mode == 7:
        model = Model_res(n_com=1,n_rec=1,com_disable=False,rec_disable=False)
        args.save_model = '1_1'
    elif args.test_mode == 8:
        model = Model_res(n_com=2,n_rec=2,com_disable=False,rec_disable=False)
        args.save_model = '2_2'
    elif args.test_mode == 9:
        model = Model_res(n_com=3,n_rec=3,com_disable=False,rec_disable=False)
        args.save_model = '3_3'
    print('test mode: {}; model name: {}'.format(args.test_mode, args.save_model))

    x_train, _ = cifar(args.train_dataset)
    succ_att, fmodel, transformer = attack(args.Linfinity)

    # Batch read attack images
    print("Preparing attack images batch.")
    attack_total = None
    true_label_list = []
    name_list = []

    for file_name, true_label, adv_label, adv_image in succ_att:
        adv_image = np.array([adv_image])
        if attack_total is None:
            attack_total = adv_image
        else:
            attack_total = np.concatenate((attack_total, adv_image), axis=0)

        name_list.append(file_name)
        true_label_list.append(true_label)

    data = tf.placeholder(tf.float32, shape=[None] + [None,None,3], name = 'data')
    is_training = tf.placeholder(tf.bool, name='is_training')
    global_steps = tf.Variable(0, trainable=False)

    placeholders = {
        'data': data,
        'is_training': is_training,
        'global_steps': global_steps,
    }

    noisy_x = data
    noisy_x = tf.clip_by_value(noisy_x,clip_value_max=1.,clip_value_min=0.)

    linear_code = model.res_com(noisy_x, is_training)

    # add gaussian before sigmoid to encourage binary code
    noisy_code = linear_code - tf.random_normal(stddev=args.noise_dev,shape=tf.shape(linear_code))
    binary_code = tf.nn.sigmoid(noisy_code)
    y =  model.res_rec(binary_code, is_training)
    binary_code_test = tf.cast(binary_code > args.binary_threshold, tf.float32)
    y_test = model.res_rec(binary_code_test, is_training)

    # optimization
    loss = tf.reduce_mean((y-noisy_x)**2) + (tf.reduce_mean(binary_code**2)) * args.lambd

    # learning rate
    if args.lr_mode == 0:
        # constant
        lr = 0.001
    elif args.lr_mode == 1:
        # constant decay
        iter_total = x_train.shape[0] // args.batch_size * args.n_epoch
        boundaries = [int(iter_total*0.25), int(iter_total*0.75), int(iter_total*0.9)]
        values = [0.01, 0.001, 0.0005, 0.0001]
        lr = tf.train.piecewise_constant_decay(global_steps, boundaries, values)
    elif args.lr_mode == 2: # TBD
        # exponential decay
        iter_total = x_train.shape[0] // args.batch_size * args.n_epoch
        lr_start = 0.01
        lr = tf.train.exponential_decay(lr_start, global_steps, iter_total // 100, 0.96, staircase=True)

    opt = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss)

    # tensorboard
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', lr)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/' + args.save_model + '@' + time_stamp, tf.get_default_graph())

    # save
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    save_model_dir = os.path.join('./models', args.save_model)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # create a session
    with tf.Session() as sess:
        # # restore checkpoints
        # ckpt = tf.train.get_checkpoint_state('./normal')
        # saver.restore(sess, ckpt.model_checkpoint_path)
        # print("Model restored from: %s" % ('./models/' + 'normal'))
        sess.run(tf.global_variables_initializer()) # init all variables

        np.random.shuffle(x_train)
        length = len(x_train)
        global_cnt = 0
        for epoch in range(args.n_epoch):
            for i in range(0, length, args.batch_size):
                global_cnt += 1

                # train
                mini_batch = x_train[i : i + args.batch_size]
                feed_dict = {
                    placeholders['data']: mini_batch,
                    placeholders['is_training']: True,
                    placeholders['global_steps']: global_cnt,
                }
                _, train_loss, merged_summary = sess.run([train_op, loss, merged], feed_dict=feed_dict)

                print("epoch: %d global_cnt: %d loss: %.3f" % (epoch, global_cnt, train_loss))
                train_writer.add_summary(merged_summary, global_cnt)

                # test
                if global_cnt % args.test_interval == 0:
                    succ_num = 0
                    # batch read attack images
                    for j in range(0, attack_total.shape[0]):
                        attack_img = attack_total[j][np.newaxis, ...]
                        true_label = true_label_list[j]

                        feed_dict = {
                            placeholders['data']: attack_img,
                            placeholders['is_training']: False,
                            placeholders['global_steps']: global_cnt,
                        }

                        bct, img_clean, bc, rec_bc = sess.run([binary_code_test, y_test, binary_code, y], feed_dict=feed_dict)

                        # cv2.imwrite('clean_images/' + name_list[j], img_clean[0, :, :, :] * 255) # (224, 224, 3)

                        img_clean = transformer(img_clean[0]).numpy()  # (3, 224, 224), float
                        label_clean_pred = np.argmax(fmodel.predictions(img_clean))

                        if true_label == label_clean_pred:
                            succ_num = succ_num + 1

                    succ_rate = succ_num / attack_total.shape[0]
                    print('Accuracy is %.3f after defending' % (succ_rate))
                    with open('./logs/' + args.save_model + '@' + time_stamp + '.txt' , 'a+') as f:
                        f.write('epoch: % d global_cnt: % d succ_num: %d succ_rate: %f \n' % (epoch, global_cnt, succ_num, succ_rate))

            save_path = saver.save(sess, os.path.join(save_model_dir, 'epoch-{}'.format(epoch)), global_step=global_cnt)
            print("Model saved in path: %s" % save_path)

    total_time = time.time() - start_time
    print('Total runtime %d hours %d minutes %.3f seconds' % (int(total_time / 3600), int(total_time % 3600 / 60), total_time % 60))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

        