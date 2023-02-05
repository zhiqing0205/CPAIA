# Multi-Task Consistent Preservation Adversarial Information Aggregation Network
![CPAIA-model](http://cdn.ziuch.cn/cross-modal-retrieval/CPAIA/CPAIA-model-2.2-1.png)

## Introduction
This is the MindSpore code implementation of an end-to-end cross-modal retrieval framework: a Multi-Task Consistent Preservation Adversarial Information Aggregation Network (CPAIA)

## Requirements
Install all required python dependencies:
```shell
pip install -r requirements.txt
```

## Dataset
We use the [XMedia](http://59.108.48.34/tiki/XMediaNet/xmedia_new.html?1217_1#Xmedia) dataset for our experiments.

```shell
git clone https://github.com/zhiqing0205/CPAIA.git
cd CPAIA
python main.py --dataset=xmedia --batchSize=64 --epoch=30 --device=GPU
```
