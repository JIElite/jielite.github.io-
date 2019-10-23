---
layout: post
title: Debias in Word Embedding
mathjax: true
---

## 前言

在處理word embedding的過程中，我們常常利用word embedding $e_i - e_j$ 的方式計算出他們embedding的差距。因為，embedding可以想成是Machine Learning從大量語料庫(corpus)學習知識後再降維到一個空間的結果。在embedding space中的距離往往可以顯現不同字(word)之間的差距，並且可以利用 $e_i - e_j$ == $e_k - e_l$的方式去判別embedding pair的相似程度。

但是，我們也可以從word embedding觀察到一些現象：性別偏見，種族偏見。


##  Reference:
[Paper: Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)