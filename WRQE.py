import numpy as np
import tensorflow as tf
import enh_networks
from scipy import misc
import argparse
import os
import my_ssim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--path_bin", default='BasketballPass_com_slow_MS-SSIM_8')
parser.add_argument("--path_raw", default='BasketballPass')
parser.add_argument("--frame", type=int, default=101)
parser.add_argument("--GOP", type=int, default=10)
parser.add_argument("--mode", default='MS-SSIM')
parser.add_argument("--l", type=int, default=8)

args = parser.parse_args()

bits = np.load(args.path_bin + '/bits.npy')
quality = np.load(args.path_bin + '/quality.npy')

img_get_size = misc.imread(args.path_bin + '/f001.png')

batch_size = 1
Height = np.size(img_get_size, 0)
Width = np.size(img_get_size, 1)
Channel = 3

step = args.GOP + 1
relu = 1
kernel = [5, 5]
filter_num = 24
CNNlayer = 5
gops = np.int((args.frame - 1)/args.GOP)

def cal_ssim(F0, F1):

    A = F0

    ssim = (my_ssim.msssim(F0[:, :, 0], F1[:, :, 0]) +
         my_ssim.msssim(F0[:, :, 1], F1[:, :, 1]) +
         my_ssim.msssim(F0[:, :, 2], F1[:, :, 2]))/3.0

    return ssim

def norm(A):
    m1 = max(A[:])
    m2 = min(A[:])

    A = (A - m2) / (m1 - m2)

    return A


def generate_weight(x_out_org, force_one):

    if force_one:

        _, x_out_s, _ = tf.split(x_out_org, [1, 9, 1], axis=-1)
        ones = tf.ones([batch_size, 1])
        x_out = tf.concat([ones, x_out_s, ones], axis=-1)

    else:

        x_out = x_out_org

    u_weight = tf.expand_dims(x_out, axis=-1)
    u_weight = tf.expand_dims(u_weight, axis=-1)
    u_weight = tf.expand_dims(u_weight, axis=-1)

    u_weight = tf.tile(u_weight, [1, 1, Height, Width, 1])

    f_weight = tf.ones([batch_size, step, Height, Width, 1]) - u_weight

    return f_weight, u_weight


def sig(x):

    a = tf.get_variable('W/a', initializer=5.0)
    y = 1 / (1 + tf.exp(-x / a))

    return y


config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

x1 = tf.placeholder(tf.float32, [batch_size, step, Height, Width, Channel])  # raw
x2 = tf.placeholder(tf.float32, [batch_size, step, Height, Width, Channel])  # compressed

x_1 = tf.placeholder(tf.float32, [batch_size, step, 10])  # feat

with tf.variable_scope("LSTM_1D"):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)
    x3, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, x_1, dtype=tf.float32)

    x33 = tf.concat([x3[0], x3[1]], axis=-1)

    x_out = tf.squeeze(enh_networks.dense(x33, step), squeeze_dims=-1)

if args.mode == 'PSNR':
    force_one = 1
else:
    force_one = 0

forget, update = generate_weight(sig(x_out), force_one=force_one)

with tf.variable_scope("ConvLSTM_2D"):

    if args.mode == 'PSNR':
        outputs = enh_networks.net_bi_wcell(x2, forget, update, step, Height, Width, filter_num, kernel, relu, CNNlayer,
                                        peephole=False, scale=False)
    elif args.mode == 'MS-SSIM':
        outputs = enh_networks.net_bi_wcell_ssim(x2, forget, update, step, Height, Width, filter_num, kernel, relu, CNNlayer,
                                       peephole=False, scale=False)

save_path = './HLVC_model/WRQE/'
saver = tf.train.Saver(max_to_keep=None)
saver.restore(sess, save_path=save_path + 'model_l' + str(args.l) + '.ckpt')

quality_ave = np.average(quality)

quality_com = np.zeros([args.frame])
quality_enh = np.zeros([args.frame])
update_g = np.zeros([args.frame])

if args.path_raw is not None:
    path_raw = args.path_raw + '/'
    frame_raw = np.zeros([batch_size, step, Height, Width, Channel])
frame_com = np.zeros([batch_size, step, Height, Width, Channel])
frame_fea = np.zeros([batch_size, step, 10])

for g in range(gops):

    quality_gop = quality[g * (step - 1): g * (step - 1) + step]
    bits_gop = bits[g * (step - 1): g * (step - 1) + step]

    quality_gop = np.concatenate((quality_gop[0:1], quality_gop[0:1], quality_gop, quality_gop[-1:], quality_gop[-1:]))
    bits_gop = np.concatenate((bits_gop[0:1], bits_gop[0:1], bits_gop, bits_gop[-1:], bits_gop[-1:]))

    for s in range(step):
        if args.path_raw is not None:
            frame_raw[0, s, :, :, :] = misc.imread(path_raw + 'f' + str(10 * g + s + 1).zfill(3) + '.png')
        frame_com[0, s, :, :, :] = misc.imread(args.path_bin + '/f' + str(10 * g + s + 1).zfill(3) + '.png')
        frame_fea[0, s, 0:5] = norm(quality_gop[s: s + 5])
        frame_fea[0, s, 5:10] = norm(bits_gop[s: s + 5])

    Y_enhanced = sess.run(outputs + x2, feed_dict={x2: frame_com / 255.0, x_1: frame_fea})

    print('Enhancing GOP:', g)

    for s in range(step):

        misc.imsave(args.path_bin + '/f' + str(10 * g + s + 1).zfill(3) + '_enh.png',
                    np.uint8(np.clip(Y_enhanced[0, s], 0, 1) * 255.0))

        if args.path_raw is not None:
            if args.mode == 'PSNR':
                # mse_com = np.mean(np.power(np.subtract(frame_raw[0, s] / 255.0, frame_com[0, s] / 255.0), 2.0))
                mse_enh = np.mean(np.power(np.subtract(frame_raw[0, s] / 255.0, Y_enhanced[0, s]), 2.0))

                # quality_com[10 * g + s] = 10.0 * np.log(1.0 / mse_com) / np.log(10.0)
                quality_enh[10 * g + s] = 10.0 * np.log(1.0 / mse_enh) / np.log(10.0)

            elif args.mode == 'MS-SSIM':
                # quality_com[10 * g + s] = cal_ssim(frame_raw[0, s], frame_com[0, s])
                quality_enh[10 * g + s] = cal_ssim(frame_raw[0, s], 255.0 * np.clip(Y_enhanced[0, s], 0, 1))

if args.path_raw is not None:
    print('Average ' + args.mode + ' (after WRQE) =', np.mean(quality_enh))

    # print(np.mean(quality_com))
    # print(np.mean(quality_enh))
    #
    # print(np.mean(quality_enh) - np.mean(quality_com))


