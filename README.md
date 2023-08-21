# Multi-Task Consistent Preservation Adversarial Information Aggregation Network
![CPAIA-model](https://img.ziuch.top/i/2023/08/21/CPAIA-model-2.2-2.png)
## Introduction
This framework is our proposed CPAIA, an end-to-end framework containing image and text sub-networks. There are three steps, the first step is feature extraction, where the feature vectors of different modalities are extracted by VGG and BOW respectively. The second step is representation separation, where the feature vectors are separated into mode-private and mode-shared components by means of a representation separation module (RS). The final step is a multi-task adversarial learning module (MA) to generate a discriminative common subspace.

## Requirements
Install all required python dependencies:
```shell
pip install -r requirements.txt
```

## Dataset
### Wikipedia:
[website](http://www.svcl.ucsd.edu/projects/crossmodal/) [link](https://file.ziuch.top/s/lt34gk)

### Nuswide：
[website](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) [link](https://file.ziuch.top/s/wzbaad)


### XMedia
[website](http://59.108.48.34/tiki/XMediaNet/)

**link：Unavailable (file only available for staff to apply, please contact the corresponding author of this article for assistance if needed)**

## Training
```shell
python main.py --dataset=xmedia --batchSize=64 --epoch=30 --device=GPU
```

## Acknowledgement
This code is based on [MindSpore](https://www.mindspore.cn/).
