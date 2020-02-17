# HighOrderInteraction-Keras

这个代码仓库使用 **Keras** 框架实现了多种引入**高阶交叉**的**推荐系统模型**，其中包含的模型有：**DCN**, **xDeepFM** 等等。除了模型实现，还附带了简化的应用程序。

## 向导

1. [环境](#环境)
2. [使用说明](#使用说明)
3. [模型](#模型)
    1. [DCN](#1-dcn)
    2. [xDeepFM](#2-xdeepfm)
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

### 2 xDeepFM

### 未完待续……

## 参考文献

1. Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings of the ADKDD'17. 2017: 1-7.
2. Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions for recommender systems[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1754-1763.