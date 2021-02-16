import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--path_bin", default='BasketballPass_com_slow_PSNR_1024')
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

# path = args.path + '/'
path_com = args.path_bin + '/'

quality_frame = np.zeros([args.frame])
bits_frame = np.zeros([args.frame])

batch_size = 1
Channel = 3

with open(path_com + '/select.bin', "rb") as ff:
    select_frame = np.frombuffer(ff.read(), dtype=np.uint8)

f = 0

if args.mode == 'PSNR':
    os.system('bpgdec ' + path_com + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
elif args.mode == 'MS-SSIM':
    os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_com + str(f + 1).zfill(3) + '.bin'
              + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

with open(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin', "rb") as ff:
    quality = np.frombuffer(ff.read(), dtype=np.float32)

bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') \
       + os.path.getsize(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin')

print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
quality_frame[f] = quality[0]
bits_frame[f] = bits * 8

for g in range(np.int(np.ceil((args.frame-1)/args.GOP))):

    # I frame

    f = (g + 1) * args.GOP

    if args.mode == 'PSNR':
        os.system('bpgdec ' + path_com + str(f + 1).zfill(3) + '.bin -o ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')
    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_com + str(
            f + 1).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

    with open(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin', "rb") as ff:
        quality = np.frombuffer(ff.read(), dtype=np.float32)

    bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') \
           + os.path.getsize(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin')

    print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
    quality_frame[f] = quality[0]
    bits_frame[f] = bits * 8

    # 2ndlayer

    f = g * args.GOP + args.GOP//2

    if select_frame[f] == 1:

        os.system(args.python_path + ' HLVC_layer2_P-frame_decoder.py --ref '
                  + path_com + 'f' + str(g * args.GOP + 1).zfill(3) + '.png'
                  + ' --com ' + path_com + 'f' + str(f + 1).zfill(3) + '.png'
                  + ' --bin ' + path_com + str(f + 1).zfill(3) + '.bin'
                  + ' --mode ' + args.mode
                  + ' --l ' + str(4 * args.l))

        with open(path_com + str(f + 1).zfill(3) + '.bin', "rb") as ff:
            quality = np.frombuffer(ff.read(4), dtype=np.float32)

        bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin')
        print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
        quality_frame[f] = quality[0]
        bits_frame[f] = bits * 8

    elif select_frame[f] == 2:

        os.system(args.python_path + ' HLVC_layer2_P-frame_decoder.py --ref '
                  + path_com + 'f' + str((g + 1) * args.GOP + 1).zfill(3) + '.png'
                  + ' --com ' + path_com + 'f' + str(f + 1).zfill(3) + '.png'
                  + ' --bin ' + path_com + str(f + 1).zfill(3) + '.bin'
                  + ' --mode ' + args.mode
                  + ' --l ' + str(4 * args.l))

        with open(path_com + str(f + 1).zfill(3) + '.bin', "rb") as ff:
            quality = np.frombuffer(ff.read(4), dtype=np.float32)

        bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin')
        print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
        quality_frame[f] = quality[0]
        bits_frame[f] = bits * 8

    elif select_frame[f] == 3:
        os.system(args.python_path + ' HLVC_layer2_B-frame_decoder.py --ref_1 '
                  + path_com + 'f' + str(g * args.GOP + 1).zfill(3) + '.png'
                  + ' --ref_2 ' + path_com + 'f' + str((g + 1) * args.GOP + 1).zfill(3) + '.png'
                  + ' --com ' + path_com + 'f' + str(f + 1).zfill(3) + '.png'
                  + ' --bin ' + path_com + str(f + 1).zfill(3) + '.bin'
                  + ' --mode ' + args.mode
                  + ' --l ' + str(4 * args.l))

        with open(path_com + str(f + 1).zfill(3) + '.bin', "rb") as ff:
            quality = np.frombuffer(ff.read(4), dtype=np.float32)

        bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin')
        print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
        quality_frame[f] = quality[0]
        bits_frame[f] = bits * 8

    elif select_frame[f] == 4:

        if args.mode == 'PSNR':
            os.system('bpgdec ' + path_com + str(f + 1).zfill(3) + '.bin -o '
                      + path_com + 'f' + str(f + 1).zfill(3) + '.png')
        elif args.mode == 'MS-SSIM':
            os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path ' + path_com + str(
                f + 1).zfill(3) + '.bin'
                      + ' --recon_path ' + path_com + 'f' + str(f + 1).zfill(3) + '.png')

        with open(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin', "rb") as ff:
            quality = np.frombuffer(ff.read(), dtype=np.float32)

        bits = os.path.getsize(path_com + str(f + 1).zfill(3) + '.bin') \
               + os.path.getsize(path_com + 'quality_' + str(f + 1).zfill(3) + '.bin')

        print('Decoded Frame', f + 1, args.mode + ' (before WRQE) =', quality[0])
        quality_frame[f] = quality[0]
        bits_frame[f] = bits * 8

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

            if select_frame[f] < 4:
                aroundlayer = 2
            elif select_frame[f] == 4:
                aroundlayer = 1 # if the second layer is intra-compressed,
                                # use the model for around layer 1

        if repeat == 2:

            f_ref = g * args.GOP + 1 + args.GOP // 2
            f_tar1 = f_ref + 1
            f_tar2 = f_ref + 2

            if select_frame[f] < 4:
                aroundlayer = 2
            elif select_frame[f] == 4:
                aroundlayer = 1

        if repeat == 3:

            f_ref = (g + 1) * args.GOP + 1
            f_tar1 = f_ref - 1
            f_tar2 = f_ref - 2
            aroundlayer = 1


        if select_frame[f_tar1 - 1] == 1:
            os.system(args.python_path + ' HLVC_layer3_P-frame_decoder.py --ref '
                      + path_com + 'f' + str(f_ref).zfill(3) + '.png'
                      + ' --com ' + path_com + 'f' + str(f_tar1).zfill(3) + '.png'
                      + ' --bin ' + path_com + str(f_tar1).zfill(3) + '.bin'
                      + ' --mode ' + args.mode
                      + ' --l ' + str(args.l) + ' --nearlayer ' + str(aroundlayer))

            with open(path_com + str(f_tar1).zfill(3) + '.bin', "rb") as ff:
                quality = np.frombuffer(ff.read(4), dtype=np.float32)

            bits = os.path.getsize(path_com + str(f_tar1).zfill(3) + '.bin')
            print('Decoded Frame', f_tar1, args.mode + ' (before WRQE) =', quality[0])
            quality_frame[f_tar1 - 1] = quality[0]
            bits_frame[f_tar1 - 1] = bits * 8

            os.system(args.python_path + ' HLVC_layer3_P-frame_decoder.py --ref '
                      + path_com + 'f' + str(f_tar1).zfill(3) + '.png'
                      + ' --com ' + path_com + 'f' + str(f_tar2).zfill(3) + '.png'
                      + ' --bin ' + path_com + str(f_tar2).zfill(3) + '.bin'
                      + ' --mode ' + args.mode
                      + ' --l ' + str(args.l) + ' --nearlayer ' + str(aroundlayer))

            with open(path_com + str(f_tar2).zfill(3) + '.bin', "rb") as ff:
                quality = np.frombuffer(ff.read(4), dtype=np.float32)

            bits = os.path.getsize(path_com + str(f_tar2).zfill(3) + '.bin')
            print('Decoded Frame', f_tar2, args.mode + ' (before WRQE) =', quality[0])
            quality_frame[f_tar2 - 1] = quality[0]
            bits_frame[f_tar2 - 1] = bits * 8

        elif select_frame[f_tar1 - 1] == 2:
            os.system(args.python_path + ' HLVC_layer3_BP-frame_decoder.py --ref '
                      + path_com + 'f' + str(f_ref).zfill(3) + '.png'
                      + ' --com_1 ' + path_com + 'f' + str(f_tar1).zfill(3) + '.png'
                      + ' --com_2 ' + path_com + 'f' + str(f_tar2).zfill(3) + '.png'
                      + ' --bin ' + path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin'
                      + ' --mode ' + args.mode
                      + ' --l ' + str(args.l) + ' --nearlayer ' + str(aroundlayer))

            with open(path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin', "rb") as ff:
                quality_1 = np.frombuffer(ff.read(4), dtype=np.float32)
                quality_2 = np.frombuffer(ff.read(4), dtype=np.float32)

            print('Decoded Frame', f_tar1, args.mode + ' (before WRQE) =', quality_1[0])
            print('Decoded Frame', f_tar2, args.mode + ' (before WRQE) =', quality_2[0])

            bits = os.path.getsize(path_com + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin')

            quality_frame[f_tar1 - 1] = quality_1[0]
            bits_frame[f_tar1 - 1] = bits * 8 / 2

            quality_frame[f_tar2 - 1] = quality_2[0]
            bits_frame[f_tar2 - 1] = bits * 8 / 2

if args.enh == 1:

    np.save(path_com + 'quality.npy', quality_frame)
    np.save(path_com + 'bits.npy', bits_frame)

    os.system(args.python_path + ' WRQE.py --path_bin ' + path_com + ' --mode ' + args.mode +
              ' --frame ' + str(args.frame) + ' --GOP ' + str(args.GOP) + ' --l ' + str(args.l)
              + ' --path_raw ' + args.path_raw)

    os.makedirs(path_com + 'frames_HLVC', exist_ok=True)
    os.system('mv *_enh.png ' + path_com + 'frames_HLVC')

os.makedirs(path_com + 'frames_beforeWRQE', exist_ok=True)
os.system('mv *.png ' + path_com + 'frames_beforeWRQE')
