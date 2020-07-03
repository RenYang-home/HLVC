import numpy as np
import os

def compare(psnr1, psnr2, bits1, bits2, a):

    if psnr2>=psnr1 and bits2<=bits1:
        r = 2 # select the pair [psnr2, bits2]
    elif psnr2<=psnr1 and bits2>=bits1:
        r = 1 # select the pair [psnr1, bits1]
    elif psnr2>=psnr1 and bits2>bits1:

        if (psnr2 - psnr1)/(bits2 - bits1) >= a:
            r = 2
        else:
            r = 1

    elif psnr2<=psnr1 and bits2<bits1:

        if (psnr1 - psnr2) / (bits1 - bits2) >= a:
            r = 1
        else:
            r = 2

    if r == 1:
        psnr = psnr1
        bits = bits1
    elif r == 2:
        psnr = psnr2
        bits = bits2

    return psnr, bits, r


def compare_four(psnr1, psnr2, psnr3, psnr4, bits1, bits2, bits3, bits4, a):

    psnr, bits, r1 = compare(psnr1, psnr2, bits1, bits2, a)
    psnr, bits, r2 = compare(psnr, psnr3, bits, bits3, a)
    psnr, bits, r3 = compare(psnr, psnr4, bits, bits4, a)

    if r2 == 1:
        r = r1
    if r2 == 2:
        r = 3
    if r3 == 2:
        r = 4

    return psnr, bits, r


def select_four(r, f, path):

    if r == 1:
       os.system('mv ' + path + 'f' + str(f + 1).zfill(3) + '_uni_pre_cand.png '
                  + path + 'f' + str(f + 1).zfill(3) + '.png')

       os.system('mv ' + path + str(f + 1).zfill(3) + '_uni_pre_cand.bin '
                  + path + str(f + 1).zfill(3) + '.bin')

    if r == 2:
       os.system('mv ' + path + 'f' + str(f + 1).zfill(3) + '_uni_post_cand.png '
                  + path + 'f' + str(f + 1).zfill(3) + '.png')

       os.system('mv ' + path + str(f + 1).zfill(3) + '_uni_post_cand.bin '
                  + path + str(f + 1).zfill(3) + '.bin')

    if r == 3:
       os.system('mv ' + path + 'f' + str(f + 1).zfill(3) + '_bi_cand.png '
                  + path + 'f' + str(f + 1).zfill(3) + '.png')

       os.system('mv ' + path + str(f + 1).zfill(3) + '_bi_cand.bin '
                  + path + str(f + 1).zfill(3) + '.bin')

    if r == 4:
       os.system('mv ' + path + 'f' + str(f + 1).zfill(3) + '_intra_cand.png '
                  + path + 'f' + str(f + 1).zfill(3) + '.png')

       os.system('mv ' + path + str(f + 1).zfill(3) + '_intra_cand.bin '
                  + path + str(f + 1).zfill(3) + '.bin')

    os.system('rm ' + path + '*_cand*')

def select(r, f_tar1, f_tar2, path):

    if r == 1:
       os.system('mv ' + path + 'f' + str(f_tar1).zfill(3) + '_uni_cand.png '
                  + path + 'f' + str(f_tar1).zfill(3) + '.png')

       os.system('mv ' + path + str(f_tar1).zfill(3) + '_uni_cand.bin '
                  + path + str(f_tar1).zfill(3) + '.bin')

       os.system('mv ' + path + 'f' + str(f_tar2).zfill(3) + '_uni_cand.png '
                 + path + 'f' + str(f_tar2).zfill(3) + '.png')

       os.system('mv ' + path + str(f_tar2).zfill(3) + '_uni_cand.bin '
                 + path + str(f_tar2).zfill(3) + '.bin')

    if r == 2:
       os.system('mv ' + path + 'f' + str(f_tar1).zfill(3) + '_bp_cand.png '
                  + path + 'f' + str(f_tar1).zfill(3) + '.png')

       os.system('mv ' + path + 'f' + str(f_tar2).zfill(3) + '_bp_cand.png '
                  + path + 'f' + str(f_tar2).zfill(3) + '.png')

       os.system('mv ' + path + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '_bp_cand.bin '
                 + path + str(f_tar1).zfill(3) + '_' + str(f_tar2).zfill(3) + '.bin')

    os.system('rm ' + path + '*_cand*')