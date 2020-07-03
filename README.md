# Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement

The project page for the paper:

> Ren Yang, Fabian Mentzer, Luc Van Gool and Radu Timofte, "Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement", in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. [[Paper]](https://arxiv.org/abs/2003.01966). 

Citation:
```
@inproceedings{yang2020Learning,
  title={Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement},
  author={Yang, Ren and Mentzer, Fabian and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

Contact:

Ren Yang @ ETH Zurich, Switzerland   

Email: ren.yang@vision.ee.ethz.ch

## Introduction

![ ](Figures/Introduction.png)

This paper proposes a Hierarchical Learned Video Compression (HLVC) method with three hierarchical quality layers and a recurrent enhancement network. As illustrated in Figure 1, the frames in layers 1, 2 and 3 are compressed with the highest, medium and the lowest quality, respectively. The benefits of hierarchical quality are two-fold: First, the high quality frames, which provide high quality references, are able to improve the compression performance of other frames at the encoder side; Second, because of the high correlation among neighboring frames, at the decoder side, the low quality frames can be enhanced by making use of the advantageous information in high quality frames. The enhancement improves quality without bit-rate overhead, thus improving the rate-distortion performance. For example, the frames 3 and 8 in Figure 1, which belong to layer 3, are compressed with low quality and bit-rate. Then, our recurrent enhancement network significantly improves their quality, taking advantage of higher quality frames, e.g., frames 0 and 5. As a result, the frames 3 and 8 reach comparable quality to frame 5 in layer 2, but consume much less bit-rate. Therefore, our HLVC approach achieves efficient video compression.

## Codes

We provide the codes for compressing video frame in various manners, i.e.,

- HLVC_layer2_P-frame(_decoder).py -- Long distance P-frame with high quality (layer 2)
- HLVC_layer2_B-frame(_decoder).py -- Long distance B-frame with high quality (layer 2)
- HLVC_layer3_P-frame(_decoder).py -- Short distance P-frame with low quality (layer 3)
- HLVC_layer3_BP-frame(_decoder).py -- Short distance BP-frames combination with low quality (layer 3), using the "single frame" strategy

Thay can be flexibly combined to achieve different frame structures and GOP sizes.

We also provide the demo codes for compress a video sequence, i.e., HLVC_video_fast/slow.py and HLVC_video_decoder.py

### Preperation

We feed RGB images into the our encoder. To compress a YUV video, please first convert to PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path_to_PNG/f%03d.png
```

Note that, our HLVC codes currently only supports the frames with the height and width as the multiples of 16. Therefore, when using these codes, if the height and width of frames are not the multiples of 16, please first crop frames, e.g.,

```
ffmpeg -pix_fmt yuv420p -s 1920x1080 -i Name.yuv -vframes Frame -filter:v "crop=1920:1072:0:0" path_to_PNG/f%03d.png
```

### Dependency

- Tensorflow 1.12

- Tensorflow-compression 1.0 ([Download link](https://github.com/tensorflow/compression/releases/tag/v1.0))

  (*After downloading, put the folder "tensorflow_compression" to the same directory as the codes.*)

- Pre-trained models ([Download link](https://drive.google.com/drive/folders/1JWRIp7RgZZEulrUfQAlbAnAkA6MAKRbE?usp=sharing))

  (*Download the folder "HLVC_model" to the same directory as the codes.*)

- BPG ([Download link](https://bellard.org/bpg/))  -- needed only for the PSNR model

  (*In our PSNR model, we use BPG to compress I-frames instead of training learned image compression models.*)

- Context-adaptive image compression model, Lee et al., ICLR 2019 ([Paper](https://arxiv.org/abs/1809.10452), [Model](https://github.com/JooyoungLeeETRI/CA_Entropy_Model)) -- needed only for the MS-SSIM model

  (*In our MS-SSIM model, we use Lee et al., ICLR 2019 to compress I-frames.*)

### How to use

- HLVC_layer2_P-frame(_decoder).py

```
--ref, reference frame.

--raw, the raw frame to be compressed. (only in the encoder)

--com, the path to save the compressed/decompressed frame.

--bin, the path to save/read the compressed bitstream.

--mode, select the PSNR/MS-SSIM optimized model.

--l, the lambda value. For layer 2, l = 32, 64, 128 and 256 for MS-SSIM, and l = 1024, 2048, 4096 and 8192 for PSNR.
```

For example,
```
python HLVC_layer2_P-frame.py --ref f001_com.png --raw f006.png --com f006_com.png --bin f006.bin --mode PSNR --l 4096
```
```
python HLVC_layer2_P-frame_decoder.py --ref f001_com.png --bin f006.bin --com f006_dec.png --mode PSNR --l 4096
```

- HLVC_layer2_B-frame(_decoder).py

Similar to HLVC_layer2_P-frame(_decoder).py but needs two reference frames.

For example,
```
python HLVC_layer2_B-frame.py --ref_1 f001_com.png --ref_2 f011_com.png --raw f006.png --com f006_com.png --bin f006.bin --mode PSNR --l 4096
```
```
python HLVC_layer2_B-frame_decoder.py --ref_1 f001_com.png --ref_2 f011_com.png --bin f006.bin --com f006_dec.png --mode PSNR --l 4096
```

- HLVC_layer3_P-frame(_decoder).py

The same network as HLVC_layer2_P-frame(_decoder).py. Since we use BPG to compressed I-frames for the PSNR model and BPG has different distortion features from learned compressed, we train two models for the Layer 3 frames near from Layer 1 (I-frames) and near from layer 2. That is,
```
parser.add_argument("--nearlayer", type=int, default=1, choices=[1, 2])
```
For example, in Figure 1, the frames 1, 2, 8 and 9 are near layer 1 and the frames 3, 4, 6 and 7 are near layer 2.
## Performance
### Settings
We test our HLVC approach on the JCT-VC (Classes B, C and D) and the [UVG](http://ultravideo.cs.tut.fi/#testsequences) datasets. Among them, the UVG and JCT-VC Class B are high resolution (1920 x 1080) datasets, and the JCT-VC Classes C and D have resolutions of 832 x 480 and 416 x 240, respectively. For a fair comparison with [Lu *et al.*, DVC](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf), we follow Lu *et al.*, DVC to test JCT-VC videos on the first 100 frames, and test UVG videos on all frames. Note that, the [UVG](http://ultravideo.cs.tut.fi/#testsequences) dataset has been enlarged recently. To compare with previous approaches, we only test on the original 7 videos in UVG, i.e., *Beauty*, *Bosphorus*, *HoneyBee*, *Jockey*, *ReadySetGo*, *ShakeNDry* and *YachtRide*.

In our approach, the entropy model requires each dimension to be a multiple of 16, and therefore we crop the 1920 x 1080 videos to 1920 x 1072 by cutting the bottom 8 pixels, using the following command.
```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -filter:v "crop=1920:1072:0:0" Name_crop.yuv
```
We calculate the Bjøntegaard-Delta Bit-Rate (BDBR) values with the anchor of *x265 LDP very fast*, which is implemented by the following command with Quality = 15, 19, 23, 27 for the JCT-VC dataset, and Quality = 11, 15, 19, 23 for UVG videos (to make the bit-rate range reasonable for comparison).
```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -r Framerate  -i  Name.yuv -vframes Frame -c:v libx265 -preset veryfast -tune zerolatency -x265-params "crf=Quality:keyint=10:verbose=1" Name.mkv
```
### Results
The detailed results (bpp, PSNR and MS-SSIM values) on each video sequence are shown in [data.xlsx](/Results). The BDBR values can be calculated by the [Matlab implementation](https://www.mathworks.com/matlabcentral/fileexchange/41749-bjontegaard-metric-calculation-bd-psnr) or the [Python implementation](https://github.com/Anserw/Bjontegaard_metric). The results are shown in Table 1, where we first calculate BDBR on each sequence, and then take the average value on each dataset. Besides, the rate-distortion curves are shown below in Figure 6. It can be seen that our HLVC approach outperforms all previous learned video compression methods and the *x265 LDP very fast* anchor. The visual results of HLVC and the *x265 LDP very fast* anchor are shown in Figure 13.

![ ](Results/BDBR.png)

![ ](Results/RD_curve.png)

![ ](Results/Visual_results.png)
