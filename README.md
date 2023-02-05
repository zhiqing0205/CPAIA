# Multi-Task Consistent Preservation Adversarial Information Aggregation Network
![CPAIA-model](http://cdn.ziuch.cn/cross-modal-retrieval/CPAIA/CPAIA-model-2.2-1.png)
## Introduction
This framework is our proposed CPAIA, an end-to-end framework containing image and text sub-networks. There are three steps, the first step is feature extraction, where the feature vectors of different modalities are extracted by VGG and BOW respectively. The second step is representation separation, where the feature vectors are separated into mode-private and mode-shared components by means of a representation separation module (RS). The final step is a multi-task adversarial learning module (MA) to generate a discriminative common subspace.

## Requirements
Install all required python dependencies:
```shell
pip install -r requirements.txt
```

## Dataset
We use the [XMedia](http://59.108.48.34/tiki/XMediaNet/xmedia_new.html?1217_1#Xmedia) dataset for our experiments.

## Training
```shell
python main.py --dataset=xmedia --batchSize=64 --epoch=30 --device=GPU
```

## Acknowledgement
This code is based on [MindSpore](https://www.mindspore.cn/).