---
layout: post
title: "Numeric Computation in Machine Learning"
---

update date: 2019/11/18: softmax


## 前言
在實作數值運算相關的演算法時，由於記憶體的不足，我們會遇到`rounding error`，`overflow`，`underflow`等問題。本篇會陸續整理一些實做Machine Learning演算法時，碰到的數值運算問題。

## Softmax
我們在處理Multi-Classification問題時，常常會碰到需要將outputs轉換為probability的情況。在數學上，我們會使用的就是`softmax` function:

$$
\text{softmax}(\boldsymbol{x})_i =  \frac{\exp{(x_i)}}{\sum_{j=1}^{d} \exp{(x_j)}}
$$

![](/assets/2019-11-18-Numeric-Computation/exponential.jpg)

當`x`很大的時候，`exp(x)`會變成`inf`，如果用來直接計算softmax就會因為overflow的問題導致計算錯誤。當`x`很小的時候，`exp(x)`會變成`0.0`，導致除零問題。

有一種解決方式是：

$$
\boldsymbol{z} = \boldsymbol{x} - \max_i x_i \boldsymbol{1}
\\
\text{And then, evaluate:} \ \text{softmax}{(\boldsymbol{z})}
\\
\text{softmax}{(\boldsymbol{z})_i} = \frac{\exp(x_i - m)}{\sum \exp(x_j -m)} = \frac{\exp(x_i) / \exp(m)}{\sum\exp(x_j)/exp(m)} = \frac{\exp{(x_i)}}{\sum\exp{(x_j)}} = \text{softmax}{(\boldsymbol{x})_i}
$$

在這種情況下，不會有overflow的問題，因為`exp(largest attribute of x) = 1`。分母最少也是1，所以也不會有除零問題。