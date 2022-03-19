# Unsupervised Domain Adaptation for Nighttime Aerial Tracking (CVPR2022)

[Junjie Ye](https://jayye99.github.io/), [Changhong Fu](https://vision4robotics.github.io/authors/changhong-fu/), [Guangze Zheng](https://zhengguangze.netlify.app/), [Danda Pani Paudel](https://people.ee.ethz.ch/~paudeld/), and [Guang Chen](https://ispc-group.github.io/). Unsupervised Domain Adaptation for Nighttime Aerial Tracking. In CVPR, pages 1-10, 2022.

![featured](https://github.com/vision4robotics/UDAT/blob/main/img/featured.png)

## Overview

**UDAT** is an unsupervised domain adaptation framework for visual object tracking. This repo contains its Python implementation.

Paper (coming soon) | [NAT2021 benchmark](https://pan.baidu.com/s/14QWmugCuGUvh2diV2i9bMw?pwd=v4rr)

## Testing UDAT

### 1. Preprocessing

Before training, we need to preprocess the unlabelled training data to generate training pairs.

1. Download the proposed [NAT2021-*train* set](https://pan.baidu.com/s/14QWmugCuGUvh2diV2i9bMw?pwd=v4rr)

2. Customize the directory of the train set in `lowlight_enhancement.py` and enhance the nighttime sequences

   ```python
   cd preprocessing/
   python lowlight_enhancement.py # enhanced sequences will be saved at '/YOUR/PATH/NAT2021/train/data_seq_enhanced/'
   ```

3. Download the video saliency detection model [here](https://drive.google.com/file/d/1Fuw3oC86AqZhH5F3pko_aqAMhPtQyt6j/view?usp=sharing) and place it at `preprocessing/models/checkpoints/`.

4. Predict salient objects and obtain candidate boxes

   ``` python
   python inference.py # candidate boxes will be saved at 'coarse_boxes/' as .npy
   ```

5. Generate pseudo annotations from candidate boxes using dynamic programming

   ``` python
   python gen_seq_bboxes.py # pseudo box sequences will be saved at 'pseudo_anno/'
   ```

6. Generate cropped training patches and a JSON file for training

   ``` py
   python par_crop.py
   python gen_json.py
   ```

### 2. Train

Take UDAT-CAR for instance.

1. Apart from above target domain dataset NAT2021, you need to download and prepare source domain datasets [VID](https://image-net.org/challenges/LSVRC/2017/) and [GOT-10K](http://got-10k.aitestunion.com/downloads).

2. Download the pre-trained daytime model ([SiamCAR](https://drive.google.com/drive/folders/11Jimzxj9QONOACJBKzMQ9La6GZhA73QD?usp=sharing)/[SiamBAN](https://drive.google.com/drive/folders/17Uz3dZFOtx-uU7J4t48_nAfPXvNsQAAq?usp=sharing)) and place it at `UDAT/tools/snapshot`.

3. Start training

   ``` python
   cd UDAT/CAR
   export PYTHONPATH=$PWD
   python tools/train.py
   ```

### 3. Test
Take UDAT-CAR for instance.
1. For quick test, you can download our trained model for [UDAT-CAR](https://drive.google.com/file/d/1DccbQ4nh2rlni8RVykTNzuHXJgSvNE4G/view?usp=sharing) (or [UDAT-BAN](https://drive.google.com/file/d/1nKyzA0ohOmrvSvypM-0cCvGNo93ZvdLp/view?usp=sharing)) and place it at `UDAT/CAR/experiments/udatcar_r50_l234`.

2. Start testing

    ```python
    python tools/test.py --dataset NAT
    ```

### 4. Eval

1. Start evaluating
    ``` python
    python tools/eval.py --dataset NAT
    ```

## Demo
[![Demo video](https://res.cloudinary.com/marcomontalbano/image/upload/v1647705190/video_to_markdown/images/youtube---nB5XitC-Lk-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/-nB5XitC-Lk "Demo video")

## Reference

> @Inproceedings{Ye2022CVPR,
>
> title={{Unsupervised Domain Adaptation for Nighttime Aerial Tracking}},
>
> author={Ye, Junjie and Fu, Changhong and Zheng, Guangze and Paudel, Danda Pani and Chen, Guang},
>
> booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
>
> year={2022}, 
>
> pages={1-10} 
>
> }



### Acknowledgments

We sincerely thank the contribution of following repos: [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR), [SiamBAN](https://github.com/hqucv/siamban), [DCFNet](https://github.com/Roudgers/DCFNet), [DCE](https://github.com/Li-Chongyi/Zero-DCE), and [USOT](https://github.com/VISION-SJTU/USOT).



### Contact

If you have any questions, please contact Junjie Ye at [ye.jun.jie@tongji.edu.cn](mailto:ye.jun.jie@tongji.edu.cn) or Changhong Fu at [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn).

