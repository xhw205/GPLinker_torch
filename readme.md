## 实体关系抽取_GPLinker

### 介绍

实体关系抽取是文本结构化、构建专业知识图谱的核心step。

本算法是[GPLinker](https://kexue.fm/archives/8888)的pytorch复现(简单易懂，杜绝花里胡哨)，该方法的核心是：

+ 对输入文本 S={w1,w2,...,wn} 的编码向量以【token-pair】标记方式建模 n×n 大小的词元矩阵，进而做实体识别、实体关系抽取任务。
+ 与之相似的工作有：[TP-Linker](https://arxiv.org/abs/2010.13415)、[multi-head selection](https://arxiv.org/abs/1804.07847)、[Word-pair](https://arxiv.org/pdf/2112.10070.pdf) 等。较之传统的 BIO 序列标注、span指针网络标注方式，token-pair 建模方式现在是实体关系抽取 sota 的必备。

### 数据集

中文医疗信息处理挑战榜CBLUE 中CMeIE数据集，同样是 CHIP2020/2021 的医学实体关系抽取数据集。

### 环境

+ python 3.8.1
+ pytorch==1.8.1
+ transformer==4.9.2
+ configparser

### 预训练模型

[RoBerta-zh-large](https://drive.google.com/file/d/1yK_P8VhWZtdgzaG0gJ3zUGOKWODitKXZ/view)下载

### 运行

请把 config.ini 中对应的【paths】换为你自己的

#### train

```python
python main.py
```

#### predict

```
python predict.py
```

### 效果
![1649315367(1).jpg](https://s2.loli.net/2022/04/07/QK2wMODjS8EnYzi.png)
+ 使用医学实体关系抽取数据集，[阿里天池](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#4)在线测试F1分数【59.82%】
+ 之前复现的CasRel方法，在线F1分数为【60%】
> 注意最新的 CBLUE 打榜，需要把生成的CMeIE_test.json后缀改为jsonl，再压缩提交
+ TODO: 训练过程未根据验证集的F1分数保存最优模型，直接用的最后一个epoch的权重，有需要的自行实现就好了

