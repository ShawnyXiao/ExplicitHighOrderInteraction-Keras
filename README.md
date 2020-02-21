# HighOrderInteraction-Keras

这个代码仓库使用 **Keras** 框架实现了多种引入**显式高阶交叉**的**推荐系统模型**，其中包含的模型有：**DCN**, **xDeepFM**, **AutoInt** 等等。除了模型实现，还附带了简化的应用程序。

## 向导

1. [环境](#环境)
2. [使用说明](#使用说明)
3. [模型](#模型)
    1. [DCN](#1-dcn)
    2. [xDeepFM](#2-xdeepfm)
    3. [AutoInt](#3-autoint)
    999. [未完待续……](#未完待续)
4. [引用](#引用)

## 环境

- Python 3.7
- NumPy 1.17.2
- Tensorflow 2.0.0

## 使用说明

代码部分都位于目录 `/model` 下，每种模型有相应的目录，该目录下放置了模型代码和应用代码。

例如：DCN 的模型代码和应用代码都位于 `/model/DCN` 下，模型部分是 `dcn.py` 和 `cross_network.py`，应用部分是 `main.py`。

## 模型

### 1 DCN

<p align="center">
	<img src="image/dcn.png">
</p>

上图展示了 DCN 的模型框架，由下至上分别是：输入层，Deep Network（右半部分）与 Cross Network（左半部分），输出层。

1. **输入层**。输入特征主要包含连续型特征（Dense Feature）和离散型特征（Sparse Feature），离散型特征通过 Embedding 层嵌入成向量，并与连续型特征拼接输送给后续模块。
2. **Deep Network 与 Cross Network**。该模块的作用是将输入层传来的向量进行隐式交叉和显示交叉，学习数据中蕴含的重要信息，并将信息表征成向量输送给最后的输出层。
3. **输出层**。该层接收表征向量，矩阵变换后，通过 Sigmoid 函数进行缩放，得到最终的输出值。

本项目使用 Keras 框架实现了以下的模块和模型：

1. 基于 tf.keras.layers.Layer 类的 Cross Network 模块
2. 基于 tf.keras.Model 类的 DCN 模型

### 2 xDeepFM

### 3 AutoInt

### 未完待续……

## 参考文献

1. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)
2. [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)
3. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)