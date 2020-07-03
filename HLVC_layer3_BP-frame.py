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
parser.add_argument("--raw_1", default='raw_1.png')
parser.add_argument("--raw_2", default='raw_2.png')
parser.add_argument("--com_1", default='com_1.png')
parser.add_argument("--com_2", default='com_2.png')
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
Y1_raw_img = misc.imread(args.raw_1)
Y2_raw_img = misc.imread(args.raw_2)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
Y1_raw_img = np.expand_dims(Y1_raw_img, 0)
Y2_raw_img = np.expand_dims(Y2_raw_img, 0)

Height = np.size(Y1_raw_img, 1)
Width = np.size(Y1_raw_img, 2)

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y2_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])


with tf.variable_scope("flow_motion", reuse=False):

    flow_20, _, _, _, _, _ = motion.optical_flow(Y0_com, Y2_raw, batch_size, Height, Width)
    # Y2_warp_0 = tf.contrib.image.dense_image_warp(Y0_com_tensor, flow_20)

with tf.variable_scope("motion_compression", reuse=False):

    flow_latent = CNN_img.MV_analysis(flow_20, num_filters=args.N, M=args.M)

    entropy_mv = tfc.EntropyBottleneck()
    string_mv = entropy_mv.compress(flow_latent)
    string_mv = tf.squeeze(string_mv, axis=0)

    flow_latent_hat, MV_likelihoods = entropy_mv(flow_latent, training=False)

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

    Res_2 = Y2_raw - Y2_MC

    res2_latent = CNN_img.Res_analysis(Res_2, num_filters=args.N, M=args.M)

    entropy_res_2 = tfc.EntropyBottleneck()
    string_res2 = entropy_res_2.compress(res2_latent)
    string_res2 = tf.squeeze(string_res2, axis=0)

    res2_latent_hat, Res_2_likelihoods = entropy_res_2(res2_latent, training=False)

    Res_2_hat = CNN_img.Res_synthesis(res2_latent_hat, num_filters=args.N)

    Y2_com = tf.clip_by_value(Res_2_hat + Y2_MC, 0, 1)

with tf.variable_scope("MC_bi", reuse=False):

    Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_10_hat)
    Y1_warp_2 = tf.contrib.image.dense_image_warp(Y2_com, flow_12_hat)
    MC_input_1 = tf.concat([flow_10_hat, Y0_com, Y1_warp_0, flow_12_hat, Y2_com, Y1_warp_2], axis=-1)
    Y1_MC = MC_network.MC(MC_input_1)

with tf.variable_scope("Res_compression_1", reuse=False):

    Res_1 = Y1_raw - Y1_MC

    res1_latent = CNN_img.Res_analysis(Res_1, num_filters=args.N, M=args.M)

    entropy_res_1 = tfc.EntropyBottleneck()
    string_res1 = entropy_res_1.compress(res1_latent)
    string_res1 = tf.squeeze(string_res1, axis=0)

    res1_latent_hat, Res_1_likelihoods = entropy_res_1(res1_latent, training=False)

    Res_1_hat = CNN_img.Res_synthesis(res1_latent_hat, num_filters=args.N)

    Y1_com = tf.clip_by_value(Res_1_hat + Y1_MC, 0, 1)

if args.mode == 'PSNR':
    train_mse_1 = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
    quality_1 = 10.0*tf.log(1.0/train_mse_1)/tf.log(10.0)
    train_mse_2 = tf.reduce_mean(tf.squared_difference(Y2_com, Y2_raw))
    quality_2 = 10.0*tf.log(1.0/train_mse_2)/tf.log(10.0)
elif args.mode == 'MS-SSIM':
    quality_1 = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))
    quality_2 = tf.math.reduce_mean(tf.image.ssim_multiscale(Y2_com, Y2_raw, max_val=1))


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


com_frame_1, com_frame_2, string_MV, string_Res_1, string_Res_2, quality_com1, quality_com2 \
    = sess.run([Y1_com, Y2_com, string_mv, string_res1, string_res2, quality_1, quality_2],
               feed_dict={Y0_com: Y0_com_img / 255.0,
                          Y1_raw: Y1_raw_img / 255.0,
                          Y2_raw: Y2_raw_img / 255.0})

with open(args.bin, "wb") as ff:
    ff.write(np.array(quality_com1, dtype=np.float32).tobytes())
    ff.write(np.array(quality_com2, dtype=np.float32).tobytes())
    ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
    ff.write(string_MV)
    ff.write(np.array(len(string_Res_1), dtype=np.uint16).tobytes())
    ff.write(string_Res_1)
    ff.write(string_Res_2)

misc.imsave(args.com_1, np.uint8(np.round(com_frame_1[0] * 255.0)))
misc.imsave(args.com_2, np.uint8(np.round(com_frame_2[0] * 255.0)))

# bpp_1 = (6 + len(string_MV)/2 + len(string_Res_1)) * 8 / Height / Width
# bpp_2 = (6 + len(string_MV)/2 + len(string_Res_2)) * 8 / Height / Width

bpp = (12 + len(string_MV) + len(string_Res_1) + len(string_Res_2)) * 8 / Height / Width

print(args.mode + ' = ' + str(quality_com1))#, 'bpp = ' + str(bpp_1))
print(args.mode + ' = ' + str(quality_com2), 'Average bpp = ' + str(bpp/2))

