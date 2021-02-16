import argparse
import numpy as np
import os
from scipy import misc
from ms_ssim_np import MultiScaleSSIM
from Compare_select import compare
from Compare_select import select
from Compare_select import compare_four
from Compare_select import select_four

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=101)
parser.add_argument("--GOP", type=int, default=10, choices=[10])
# Do not change the GOP size, this demo only supports GOP = 10. Other GOPs need to modify this code.
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--python_path", default='python')
parser.add_argument("--CA_model_path", default='CA_EntropyModel_Test')
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--enh", type=int, default=1, choices=[0, 1])
args = parser.parse_args()

assert (args.frame % args.GOP == 1)

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

elif args.l == 8:
    I_level = 2
elif args.l == 16:
    I_level = 3
elif args.l == 32:
    I_level = 5
elif args.l == 64:
    I_level = 7

path = args.path + '/'
path_com = args.path + '_com_fast_' + args.mode  + '_' + str(args.l) + '/'

os.makedirs(path_com, exist_ok=True)

batch_size = 1
Channel = 3

F1 = misc.imread(path + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)

if (Height % 16 != 0) or (Width % 16 != 0):
    raise ValueError('Height and Width must be a mutiple of 16.')

quality_frame = np.zeros([args.frame])
bits_frame = np.zeros([args.frame])
select_frame = np.zeros([args.frame])

f = 0

if args.mode == 'PSNR':
    os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(f + 1).zfill(3) + '.png -o ' + path_com + str(f + 1).zfill(3) + '.bin -q ' + str(I_QP))
    os.system('bpgdec ' + path_com + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
elif args.mode == 'MS-SSIM':
    os.system(args.python_path + ' ' + args.CA_model_path + '/encode.py --model_type 1 --input_path ' + path + 'f' + str(f + 1).zfill(3) + '.png' +
              ' --compressed_file_path ' + path_com + str(f + 1).zfill(3) + '.bin' + ' --quality_level ' + str(I_level))
    os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_com + str(f + 1).zfill(3) + '.bin'
              + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

F0_com = misc.imread(path_com + 'f' + str(f + 1).zfill(3) + '.png')
F0_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')

if args.mode == 'PSNR':
    mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
    quality_frame[f] = 10 * np.log10(1.0 / mse)
elif args.mode == 'MS-SSIM':
    quality_frame[f] = MultiScaleSSIM(np.expand_dims(F0_com, 0),
                                      np.expand_dims(F0_raw, 0), max_val=255)

with open(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin', "wb") as ff:
    ff.write(np.array(quality_frame[f], dtype=np.float32).tobytes())

bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') \
       + os.path.getsize(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin')

bits = bits * 8

bits_frame[f] = bits / Height / Width
print('Frame', f + 1)
print(args.mode + ' (before WRQE) =', quality_frame[f], 'bpp =', bits_frame[f])

for g in range(np.int(np.ceil((args.frame-1)/args.GOP))):

    # I frame

    f = (g + 1) * args.GOP

    if args.mode == 'PSNR':
        os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(f + 1).zfill(3) + '.png -o ' + path_com + str(f + 1).zfill(
            3) + '.bin -q ' + str(I_QP))
        os.system(
            'bpgdec ' + path_com + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
    elif args.mode == 'MS-SSIM':
        os.system(
            args.python_path + ' ' + args.CA_model_path + '/encode.py --model_type 1 --input_path ' + path + 'f' + str(
                f + 1).zfill(3) + '.png' +
            ' --compressed_file_path ' + path_com + str(f + 1).zfill(3) + '.bin' + ' --quality_level ' + str(I_level))
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_com + str(
            f + 1).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

    F0_com = misc.imread(path_com + 'f' + str(f + 1).zfill(3) + '.png')
    F0_raw = misc.imread(path + 'f' + str(f + 1).zfill(3) + '.png')

    if args.mode == 'PSNR':
        mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
        quality_frame[f] = 10 * np.log10(1.0 / mse)
    elif args.mode == 'MS-SSIM':
        quality_frame[f] = MultiScaleSSIM(np.expand_dims(F0_com, 0),
                                          np.expand_dims(F0_raw, 0), max_val=255)

    with open(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin', "wb") as ff:
        ff.write(np.array(quality_frame[f], dtype=np.float32).tobytes())

    bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') \
           + os.path.getsize(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin')
    bits = bits * 8

    bits_frame[f] = bits / Height / Width
    print('Frame', f + 1)
    print(args.mode + ' (before WRQE) =', quality_frame[f], 'bpp =', bits_frame[f])

    # 2ndlayer

    f = g * args.GOP + args.GOP//2

    ## B-frame
    print('Frame', f + 1)
    os.system(args.python_path + ' HLVC_layer2_B-frame.py --ref_1 '
              + path_com + 'f' + str(g * args.GOP + 1).zfill(3) + '.png'
              + ' --ref_2 ' + path_com + 'f' + str((g + 1) * args.GOP + 1).zfill(3) + '.png'
              + ' --raw ' + path + 'f' + str(f + 1).zfill(3) + '.png'
              + ' --com ' + path_com + 'f' + str(f + 1).zfill(3) + '.png'
              + ' --bin ' + path_com + str(f + 1).zfill(3) + '.bin'
              + ' --mode ' + args.mode
              + ' --l ' + str(4 * args.l))

    bits_3 = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') * 8.0 / Height / Width
    with open(path_com + str(f + 1).zfill(3) + '.bin', "rb") as ff:
        quality_3 = np.frombuffer(ff.read(4), dtype=np.float32)

    select_frame[f] = 3
    quality_frame[f] = quality_3
    bits_frame[f] = bits_3

    # 3rdlayer

    for repeat in range(4):


        if repeat == 0:

            f_ref = g * args.GOP + 1
            f_tar1 = f_ref + 1
            f_tar2 = f_ref + 2
            aroundlayer = 1

        if repeat == 1:

            f_ref = g * args.GOP + 1 + args.GOP//2
            f_tar1 = f_ref - 1
            f_tar2 = f_ref - 2

            aroundlayer = 2


        if repeat == 2:

            f_ref = g * args.GOP + 1 + args.GOP // 2
            f_tar1 = f_ref + 1
            f_tar2 = f_ref + 2

            aroundlayer = 2


        if repeat == 3:

            f_ref = (g + 1) * args.GOP + 1
            f_tar1 = f_ref - 1
            f_tar2 = f_ref - 2
            aroundlayer = 1

        ## A pair of BP-frames
        print('Frame', f_tar1, '&', f_tar2)
        os.system(args.python_path + ' HLVC_layer3_BP-frame.py --ref '
                  + path_com + 'f' + str(f_ref).zfill(3) + '.png'
                  + ' --raw_1 ' + path + 'f' + str(f_tar1).zfill(3) + '.png'
                  + ' --raw_2 ' + path + 'f' + str(f_tar2).zfill(3) + '.png'
                  + ' --com_1 ' + path_com + 'f' + str(f_tar1).zfill(3) + '.png'
                  + ' --com_2 ' + path_com + 'f' + str(f_tar2).zfill(3) + '.png'
                  + ' --bin ' + path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin'
                  + ' --mode ' + args.mode
                  + ' --l ' + str(args.l) + ' --nearlayer ' + str(aroundlayer))

        bits_bp = os.path.getsize(path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin') * 8.0 / Height / Width
        with open(path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin', "rb") as ff:
            quality_3 = np.frombuffer(ff.read(4), dtype=np.float32)
            quality_4 = np.frombuffer(ff.read(4), dtype=np.float32)

        quality_bp = (quality_3 + quality_4) / 2.0

        select_frame[f_tar1 - 1] = 2
        select_frame[f_tar2 - 1] = 2

        quality_frame[f_tar1 - 1] = quality_3
        bits_frame[f_tar1 - 1] = bits_bp / 2

        quality_frame[f_tar2 - 1] = quality_4
        bits_frame[f_tar2 - 1] = bits_bp / 2

quality_ave = np.average(quality_frame)
bits_ave = np.average(bits_frame)

with open(path_com + 'select.bin', "wb") as ff:
    ff.write(np.array(select_frame, dtype=np.uint8).tobytes())

bits_ave += os.path.getsize(path_com + 'select.bin') * 8 / Height / Width / args.frame

print('Average ' + args.mode + ' (before WRQE) =', quality_ave, 'Average bpp =', bits_ave)

if args.enh == 1:

    np.save(path_com + 'quality.npy', quality_frame)
    np.save(path_com + 'bits.npy', bits_frame)

    os.system(args.python_path + ' WRQE.py --path_bin ' + path_com + ' --mode ' + args.mode +
              ' --frame ' + str(args.frame) + ' --GOP ' + str(args.GOP) + ' --l ' + str(args.l)
              + ' --path_raw ' + args.path)

    os.makedirs(path_com + 'frames_HLVC_fast', exist_ok=True)
    os.system('mv ' + path_com + '*_enh.png ' + path_com + 'frames_HLVC')

os.makedirs(path_com + 'frames_beforeWRQE_fast', exist_ok=True)
os.system('mv ' + path_com + '*.png ' + path_com + 'frames_beforeWRQE')
