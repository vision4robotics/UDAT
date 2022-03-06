# Unsupervised Domain Adaptation for Nighttime Aerial Tracking (CVPR2022)

[Junjie Ye](https://jayye99.github.io/), [Changhong Fu](https://vision4robotics.github.io/authors/changhong-fu/), [Guangze Zheng](https://vision4robotics.github.io/authors/guangze-zheng/), [Danda Pani Paudel](https://people.ee.ethz.ch/~paudeld/), and [Guang Chen](https://ispc-group.github.io/). Unsupervised Domain Adaptation for Nighttime Aerial Tracking. In CVPR, pages 1-10, 2022.

## Abstract

Previous advances in object tracking mostly reported on favorable illumination circumstances while neglecting performance at nighttime, which significantly impeded the development of related aerial robot applications. This work instead develops a novel unsupervised domain adaptation framework for nighttime aerial tracking (named **UDAT**). Specifically, a unique object discovery approach is provided to generate training patches from raw nighttime tracking videos. To tackle the domain discrepancy, we employ a Transformer-based bridging layer post to the feature extractor to align image features from both domains. With a Transformer day/night feature discriminator, the daytime tracking model is adversarially trained to track at night. Moreover, we construct a pioneering benchmark namely **NAT2021** for unsupervised domain adaptive nighttime tracking, which comprises a test set of 180 manually annotated tracking sequences and a train set of over 285k unlabelled nighttime tracking frames. Exhaustive experiments demonstrate the robustness and domain adaptability of the proposed framework in nighttime aerial tracking.

![featured](https://github.com/vision4robotics/UDAT/blob/main/img/featured.png)



**The code of UDAT and the NAT2021 benchmark will be released here soon~**



## Reference

> @Inproceedings{Ye2022CVPR,
>   title={{Unsupervised Domain Adaptation for Nighttime Aerial Tracking}},
>   author={Ye, Junjie and Fu, Changhong and Zheng, Guangze and Paudel, Danda Pani and Chen, Guang},
>   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
>   year={2022}, 
>   pages={1-10} 
> }

