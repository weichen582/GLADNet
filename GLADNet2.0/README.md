# GLADNet

This is a Tensorflow implantation of GLADNet

GLADNet: Low-Light Enhancement Network with Global Awareness. In [FG'18](https://fg2018.cse.sc.edu/index.html) Workshop [FOR-LQ 2018](http://staff.ustc.edu.cn/~dongeliu/forlq2018/index.html) <br>
[Wenjing Wang*](https://daooshee.github.io/website/), [Chen Wei*](https://weichen582.github.io/), [Wenhan Yang](https://flyywh.github.io/), [Jiaying Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html). (* indicates equal contributions)<br>

[Paper](http://www.icst.pku.edu.cn/F/course/icb/Pub%20Files/2018/wwj_fg2018.pdf), [Project Page](https://daooshee.github.io/fgworkshop18Gladnet/)

![Teaser Image](https://github.com/daooshee/fgworkshop18Gladnet/blob/master/images/fg-1478.jpg)

## Requirements ##
1. Python
2. Tensorflow >= 1.3.0
3. numpy, PIL

## Testing  Usage ##
To quickly test your own images with our model, you can just run through
```shell
python main.py 
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.5 \                         # gpu memory usage
    --phase=test \
    --test_dir=/path/to/your/test/dir/ \
    --save_dir=/path/to/save/results/ \
```
## Training Usage ##
First, download training data set from [our project page](https://daooshee.github.io/fgworkshop18Gladnet/). Save training pairs of our LOL dataset under `./data/train/low/`, and synthetic pairs under `./data/train/normal/`.
Then, start training by 
```shell
python main.py
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.8 \                         # gpu memory usage
    --phase=train \
    --epoch=50 \                           # number of training epoches
    --batch_size=8 \
    --patch_size=384 \                       # size of training patches
    --base_lr=0.001 \                      # initial learning rate for adm
    --eval_every_epoch=5 \                 # evaluate and save checkpoints for every # epoches
    --checkpoint_dir=./checkpoint           # if it is not existed, automatically make dirs
    --sample_dir=./sample                   # dir for saving evaluation results during training
 ```
 
 ## Experiment Results ##
 #### Subjective Results ####
 ![Subjective Result](https://github.com/daooshee/fgworkshop18Gladnet/blob/master/images/result-1532-2.jpg)
 #### Objective Results ####
We use the [Naturalness Image Quality Evaluator (NIQE)](https://ieeexplore.ieee.org/document/6353522) no-reference image quality score for quantitative comparison. NIQE compares images to a default model computed from images of natural scenes. A smaller score indicates better perceptual quality.
 
| Dataset | DICM | NPE | MEF | Average |
| ------ | ------ | ------ | ------ | ------ |
| MSRCR | 3.117 | 3.369 | 4.362 | 3.586 |
| LIME | 3.243 | 3.649 | 4.745 | 3.885 |
| DeHZ | 3.608 | 4.258 | 5.071 | 4.338 | 
| SRIE | 2.975 | <b>3.127</b> | 4.042 | 3.381 |
| <b>GLADNet</b> | <b>2.761</b> | 3.278 | <b>3.468</b> | <b>3.184</b> |
 #### Computer Vision Application ####
 We test several real low-light images and their corresponding enhanced results on [Google Cloud Visio API](https://cloud.google.com/vision/). GLADNet helps it to identify the objects in this image.
  <br>
  <br>
  ![APP1](https://raw.githubusercontent.com/daooshee/fgworkshop18Gladnet/master/images/app1-1546-2.jpg)
  <br>
  <br>
  ![APP2](https://raw.githubusercontent.com/daooshee/fgworkshop18Gladnet/master/images/app2-1482.jpg)

 ## Citation ##
 ```
@inproceedings{wang2018gladnet,
  title={GLADNet: Low-Light Enhancement Network with Global Awareness},
  author={Wang, Wenjing and Wei, Chen and Yang, Wenhan and Liu, Jiaying},
  booktitle={Automatic Face \& Gesture Recognition (FG 2018), 2018 13th IEEE International Conference},
  pages={751--755},
  year={2018},
  organization={IEEE}
}
```

## Related Follow-Up Work ##
Deep Retinex Decomposition: <b>Deep Retinex Decomposition for Low-Light Enhancement</b>. Chen Wei*, Wenjing Wang*, Wenhan Yang, Jiaying Liu. (* indicates equal contributions) In BMVC'18 (Oral Presentation) [Website](https://daooshee.github.io/BMVC2018website/) [Github](https://github.com/weichen582/RetinexNet)
 
