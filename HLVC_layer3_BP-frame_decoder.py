import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from scipy import misc
import CNN_img
import motion
import MC_network
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref", default='ref.png')
# parser.add_argument("--raw_1", default='raw_1.png')
# parser.add_argument("--raw_2", default='raw_2.png')
parser.add_argument("--com_1", default='dec_1.png')
parser.add_argument("--com_2", default='dec_2.png')
parser.add_argument("--bin", default='bits_BP.bin')
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--nearlayer", type=int, default=1, choices=[1, 2])
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

batch_size = 1
Channel = 3

Y0_com_img = misc.imread(args.ref)
# Y1_raw_img = misc.imread(args.raw_1)
# Y2_raw_img = misc.imread(args.raw_2)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
# Y1_raw_img = np.expand_dims(Y1_raw_img, 0)
# Y2_raw_img = np.expand_dims(Y2_raw_img, 0)

Height = np.size(Y0_com_img, 1)
Width = np.size(Y0_com_img, 2)

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
# Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
# Y2_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

string_mv_tensor = tf.placeholder(tf.string, [])
string_res1_tensor = tf.placeholder(tf.string, [])
string_res2_tensor = tf.placeholder(tf.string, [])


with tf.variable_scope("motion_compression", reuse=False):

    entropy_mv = tfc.EntropyBottleneck(dtype=tf.float32)
    flow_latent_hat = entropy_mv.decompress(
        tf.expand_dims(string_mv_tensor, 0), [Height // 16, Width // 16, args.M], channels=args.M)

    flow_20_hat = CNN_img.MV_synthesis(flow_latent_hat, num_filters=args.N)

with tf.variable_scope("motion_estimation", reuse=False):

    flow_02_hat = motion.tf_inverse_flow(flow_20_hat, batch_size, Height, Width)
    flow_01_hat = 0.5 * flow_02_hat
    flow_10_hat = motion.tf_inverse_flow(flow_01_hat, batch_size, Height, Width)

    flow_21_hat = 0.5 * flow_20_hat
    flow_12_hat = motion.tf_inverse_flow(flow_21_hat, batch_size, Height, Width)

with tf.variable_scope("MC_uni", reuse=False):

    Y2_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_20_hat)
    MC_input_2 = tf.concat([flow_20_hat, Y0_com, Y2_warp], axis=-1)
    Y2_MC = MC_network.MC(MC_input_2)

with tf.variable_scope("Res_compression_2", reuse=False):

    entropy_res_2 = tfc.EntropyBottleneck(dtype=tf.float32)
    res2_latent_hat = entropy_res_2.decompress(
        tf.expand_dims(string_res2_tensor, 0), [Height // 16, Width // 16, args.M], channels=args.M)

    Res_2_hat = CNN_img.Res_synthesis(res2_latent_hat, num_filters=args.N)

    Y2_com = tf.clip_by_value(Res_2_hat + Y2_MC, 0, 1)

with tf.variable_scope("MC_bi", reuse=False):

    Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_10_hat)
    Y1_warp_2 = tf.contrib.image.dense_image_warp(Y2_com, flow_12_hat)
    MC_input_1 = tf.concat([flow_10_hat, Y0_com, Y1_warp_0, flow_12_hat, Y2_com, Y1_warp_2], axis=-1)
    Y1_MC = MC_network.MC(MC_input_1)

with tf.variable_scope("Res_compression_1", reuse=False):

    entropy_res_1 = tfc.EntropyBottleneck(dtype=tf.float32)
    res1_latent_hat = entropy_res_1.decompress(
        tf.expand_dims(string_res1_tensor, 0), [Height // 16, Width // 16, args.M], channels=args.M)

    Res_1_hat = CNN_img.Res_synthesis(res1_latent_hat, num_filters=args.N)

    Y1_com = tf.clip_by_value(Res_1_hat + Y1_MC, 0, 1)

# if args.mode == 'PSNR':
#     train_mse_1 = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
#     quality_1 = 10.0*tf.log(1.0/train_mse_1)/tf.log(10.0)
#     train_mse_2 = tf.reduce_mean(tf.squared_difference(Y2_com, Y2_raw))
#     quality_2 = 10.0*tf.log(1.0/train_mse_2)/tf.log(10.0)
# elif args.mode == 'MS-SSIM':
#     quality_1 = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))
#     quality_2 = tf.math.reduce_mean(tf.image.ssim_multiscale(Y2_com, Y2_raw, max_val=1))

sess.run(tf.global_variables_initializer())
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
opt_vars = [v for v in all_vars if 'motion_estimation' not in v.name]

saver = tf.train.Saver(var_list=opt_vars, max_to_keep=None)

if args.mode == 'MS-SSIM':
    model_path = './HLVC_model/Layer3_BP-frame/' \
             'Layer3_BP_' + args.mode + '_' + str(args.l) + '/model.ckpt'

elif args.mode =='PSNR':
    model_path = './HLVC_model/Layer3_BP-frame/' \
                 'Layer3_BP_' + args.mode + '_' + str(args.l) \
                 + '_aroundlayer' + str(args.nearlayer) + '/model.ckpt'

saver.restore(sess, save_path=model_path)


with open(args.bin, "rb") as ff:
    quality_com1 = np.frombuffer(ff.read(4), dtype=np.float32)
    quality_com2 = np.frombuffer(ff.read(4), dtype=np.float32)
    mv_len = np.frombuffer(ff.read(2), dtype=np.uint16)
    string_mv = ff.read(np.int(mv_len))
    res1_len = np.frombuffer(ff.read(2), dtype=np.uint16)
    string_res1 = ff.read(np.int(res1_len))
    string_res2 = ff.read()

com_frame_1, com_frame_2 = sess.run([Y1_com, Y2_com],
               feed_dict={Y0_com: Y0_com_img / 255.0,
                          # Y1_raw: Y1_raw_img / 255.0,
                          # Y2_raw: Y2_raw_img / 255.0,
                          string_mv_tensor: string_mv,
                          string_res1_tensor: string_res1,
                          string_res2_tensor: string_res2})

misc.imsave(args.com_1, np.uint8(np.round(com_frame_1[0] * 255.0)))
misc.imsave(args.com_2, np.uint8(np.round(com_frame_2[0] * 255.0)))

bpp_1 = (6 + len(string_mv)/2 + len(string_res1)) * 8 / Height / Width
bpp_2 = (6 + len(string_mv)/2 + len(string_res2)) * 8 / Height / Width

# print('Decoded Frame 1', args.mode + ' = ' + str(quality_com1[0]), 'bpp = ' + str(bpp_1))
# print('Decoded Frame 2', args.mode + ' = ' + str(quality_com2[0]), 'bpp = ' + str(bpp_2))

