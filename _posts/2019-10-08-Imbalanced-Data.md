---
layout: post
title: "Imbalanced Data Processing, 不平衡資料處理-Draft"
mathjax: true
---
## Introduction
Imbalanced Data這個名詞對ML初學者來說是十分陌生，第一次聽到這個概念還是從前輩口中得知。這是一個在二元分類(Binary Classification)中很關鍵的問題。這篇是我的讀書筆記，也許不盡然正確，不過也很樂意分享給大家，希望可以一起交流。

我們知道二元分類問題就是要將某一個sample $x^{(i)}$ (the i-th example)分到 $\mathcal{Y}$, where $\mathcal{Y} = \{0, 1\}$。但是當negative類別佔了很高的比重時，我們對 $x_i$ 猜negative都會有很高的準確度(Accuracy)。例如：一個Dataset $\mathcal{D}$ 中有99.9%的樣本都是negative, 僅僅只有0.1%是positive。

從上面的敘述，我們可以知道至少面臨了兩種困境
1. Algorithm metrics的選擇，因為Accuracy不適合。
2. Negative samples和Positive Sameples的數量相差太多

因此，我們很直覺的可以從這兩方面下手。


## Algorithm metrics

以往我們常用的Accuracy在這種情況下判別出來的模型不是我們想要的，在imbalanced data的情境下，傾向同時考慮Recall&Precision。

下面是Confusion Matrix:

![](https://glassboxmedicine.files.wordpress.com/2019/02/confusion-matrix.png?w=768&h=432)
<center>Image credits: [measuring-performance-the-confusion-matrix](https://glassboxmedicine.com/2019/02/17/measuring-performance-the-confusion-matrix>)</center>

- Accuracy(A) = $\frac{TP+TN}{TP+FN+TF+FP}$
- Recall(R) = $\frac{TP}{TP+FN}$, 代表：我們找出Positive的樣本佔所有Positive data有多少。在Negative data作為大多數的情況下，我們怎麼猜都猜Negative，這樣Recall就會很慘，因為FN會很高。

- Precision(P) = $\frac{TP}{TP+FP}$，Precision也需要被考量，因為在商業上如果我們誤判為False Positive的比例太多，我們是要付出成本去收拾這個殘局的，例如：預測盜刷情況，如果顧客沒有盜刷我們就要做出一定補償。預測垃圾郵件，如果將正常的郵件分類到垃圾郵件，那代價可能不小。 

綜上，我們會同時考量Recall&Precision。很直覺的，但是我們不會單純用算術平均去計算，而是用調和平均(Harmonic Mean): 

$$ \Large{\frac{2}{\frac{1}{Recall} + \frac{1}{Precision}}} = 2 \cdot \large{\frac{Recall \cdot{Precision}}{Recall+Precision}}$$

這就是我們常聽到的`F1-score(F-measure)`

*Harmonic mean: <https://en.wikipedia.org/wiki/Harmonic_mean>*

- ROC
![](/assets/2019-10-08-Imbalanced-Data/roc.png)

然而我們有了適當的metrics去衡量我們的模型後，問題仍然沒有解決。因為樣本不平衡的情況依然存在，被標記為陽性(Labeled Positive)的樣本依然很少，同樣會造成演算法學習的問題。這樣樣本不平均的情況，目前通常利用重採樣(Resampling)的方式來解決。

## 重新採樣 Resampling
因為我們Data的數量相當不平衡，e.g, $\mid\mathcal{D_{majority}}\mid \gg \mid\mathcal{D_{minority}}\mid$，在這種情況下我們會想把他們的數量調整成一樣以便演算法學習，因此有兩種方式解決這個問題，而兩種處理方式各有優缺。
1. Oversampling: 產生數少量一方的data
2. Undersampling: 減少數量多那方的data



*[Q]對於特定algothim會不會遇到distribution不同的問題？*


**Oversampling**
- Random sampling: 

    Oversampling最直覺的方法就是利用原本少數類別的樣本去重複抽樣，藉此達到樣本數量平衡的情況。顯而易見，這種情況下效果不會太好，不僅如此，還會增加overfitting的機會。因為我們產生的是重複且原有的資料點，如果多次抽樣的點剛好是影響decision boundary最嚴重的點，這樣的情況就會更糟。

- SMOTE(**S**ynthetic **M**inority **O**versampling **T**echniqu**E**, 2002):
    
    為了要產生平衡的數據，除了Ramdom sampling之外，我們也會有個疑問:有沒有什麼方式是可以利用人工合成的方式去產生額外的樣本？SMOTE就是利用原有的樣本來合成新樣本藉以訓練模型的方法。SMOTE的基本思路是利用Nearest Neighbor，並且可以設定要合成少數樣本的倍率N: $\mid\mathcal{D_{minority}}\mid = 100, N=75, \mid\mathcal{D_{synthetic}}\mid = 7500$
    
    SMOTE Algorithm: 在 $\mathcal{D_{minority}}$ 中對每一個樣本$x^{(i)}$利用KNN去計算最近的k個相鄰的樣本，然後隨機抽樣出其中一個$x^{(j)}$，再進一步合成新的少數類別樣本。我們假設每個樣本點的特徵(features)有d個,  $x^{(i)}$中的每一維feature用下標表示 $x^{(i)}_d$，合成新樣本的方式如下：

    $$
    \Delta = x^{(j)} - x^{(i)} \\
    \eta = \mathbf{[\eta_1, \eta_2, ... , \eta_d]}^T \\
    \eta_k \sim random(0, 1) \\
    x^{(synthetic)} = x^{(i)} + \eta \odot \Delta \\
    \odot: \texttt{element-wise product}
    $$

    ```python
    import numpy as np

    # We denote x[i] as a current processing sample
    # x[j] is random sampled from KNN.
    # The shape for both x[i], x[j] is (d, 1)
    synthetic_sample = np.zeros_like(x[i])
    for d in range(x[i].shape[0]):
        delta = x[j][d] - x[i][d]
        eta_k = np.random.random()
        synthetic_sample = x[j] + eta*delta

    # Vectorization ver.
    sample_shape = x.shape
    delta = x[j] - x[i]
    eta = np.random.rand(*sample_shape)
    synthetic_sample = x[j] + eta*delta
    ```
    值得留意的是在原本的演算法中，對應到每一維度的 $\eta_k$ 來自不同的抽樣結果，有些可能是0.1有些是0.4...。這點是我在看論文中的演算法部份想到的疑問。不過他們在論文中給的範例有 $\eta_k$ 都是相同的情況，有興趣可話可以看原論文的 $table1$。個人是覺得，在合成新樣本的時候，利用的 $\eta$ 是相同的會比較合理些。此外，他們有提到原本的演算法再作KNN時，選擇的Neighbor可能不是屬於少數類別，我個人認為問題不大，因為容易修改，這在現成的package也都作為可選參數。

- Borderline-SMOTE
- SVMSMOTE
- ADASYN
- Generative Model: Baysian Network/GAN
  

**Undersampling**
- Tomek-Link
- NearMiss
- One-sided selection
- Informed Undersampling
- The cross validation on undersampling

**Ensemble method**

**Change loss function**
- Weighting Strategy
- Cost-sensitive Learning Algorithm

**Anomaly detection**
- 像是在偵測outliers一樣嗎？


最後，十分推薦imblearn這個package的網站，裡面可以看到許多examples: <https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/index.html>，裡面有很多處理imbalanced data的例子，從API也可以得知常用的metrics&algorithm。


## Questions
1. 為什麼Decision Tree, Random Forest不用擔心imbalance data issue?
2. 在Deep learning中，resampling依然奏效，可是deep learning需要的training data更為大量，有沒有什麼額外方法來解決這個問題？或是從模型設計的角度去思考?

## Related Research Papers
- [Active Learning for Decision-Making from Imbalanced Observational Data, ICML2019](https://arxiv.org/abs/1904.05268)
- [Cost-Sensitive Learning vs. Sampling: Which is Best for
Handling Unbalanced Classes with Unequal Error Costs?](https://storm.cis.fordham.edu/gweiss/papers/dmin07-weiss.pdf)
- [SMOTE: Synthetic Minority Over-sampling Technique, JAIR 2002](https://arxiv.org/pdf/1106.1813.pdf)



## Reference:
- [大鼻David's Perspective: 不平衡資料的二元分類](https://taweihuang.hpd.io/2018/12/28/imbalanced-data-performance-metrics/)
- [Dealing with Imbalanced Data, Tara Boyle](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
- [知乎: 数据嗨客 第6期：不平衡数据处理](https://zhuanlan.zhihu.com/p/21406238)
- [Book: The Quest for Machine Learning: 100+ Interview Quesions for Algorithm Engineer](https://www.tenlong.com.tw/products/9787115487360)
- [Python package: imblearn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)
- [Kaggle competition: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

others: latex size: http://www.sascha-frank.com/latex-font-size.html

