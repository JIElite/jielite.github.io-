---
layout: post
title: "Imbalanced Data Processing, 不平衡資料處理-Draft"
mathjax: true
---
## Introduction
Imbalanced Data這個名詞對ML初學者來說是十分陌生，第一次聽到這個概念還是從前輩口中得知。這是一個在二元分類(Binary Classification)中很關鍵的問題。這篇是我的讀書筆記，也許不盡然正確，不過也很樂意分享給大家，希望可以一起交流。

我們知道二元分類問題就是要將某一個sample \\(x^{(i)}\\) (the i-th example)分到 \\(\mathcal{Y}\\), where \\(\mathcal{Y} = \{0, 1\}\\)。但是當negative類別佔了很高的比重時，我們對 \\(x^{i}\\) 猜negative都會有很高的準確度(Accuracy)。例如：一個Dataset \\(\mathcal{D}\\) 中有99.9%的樣本都是negative, 僅僅只有0.1%是positive。

從上面的敘述，我們可以知道至少面臨了兩種困境
1. Algorithm metrics的選擇，因為Accuracy不適合。
2. Negative samples和Positive Sameples的數量相差太多

因此，我們很直覺的可以從這兩方面下手。


## Algorithm metrics

以往我們常用的Accuracy在這種情況下判別出來的模型不是我們想要的，在imbalanced data的情境下，傾向同時考慮Recall&Precision。

下面是Confusion Matrix:

![](https://glassboxmedicine.files.wordpress.com/2019/02/confusion-matrix.png?w=768&h=432)
<center>Image credits: [measuring-performance-the-confusion-matrix](https://glassboxmedicine.com/2019/02/17/measuring-performance-the-confusion-matrix>)</center>

- Accuracy(A) = \\(\frac{TP+TN}{TP+FN+TF+FP}\\)
- Recall(R) = \\(\frac{TP}{TP+FN}\\), 代表：我們找出Positive的樣本佔所有Positive data有多少。在Negative data作為大多數的情況下，我們怎麼猜都猜Negative，這樣Recall就會很慘，因為FN會很高。

- Precision(P) = \\(\frac{TP}{TP+FP}\\)，Precision也需要被考量，因為在商業上如果我們誤判為False Positive的比例太多，我們是要付出成本去收拾這個殘局的，例如：預測盜刷情況，如果顧客沒有盜刷我們就要做出一定補償。預測垃圾郵件，如果將正常的郵件分類到垃圾郵件，那代價可能不小。 

綜上，我們會同時考量Recall&Precision。很直覺的，但是我們不會單純用算術平均去計算，而是用調和平均(Harmonic Mean): 

$$ \Large{\frac{2}{\frac{1}{Recall} + \frac{1}{Precision}}} = 2 \cdot \large{\frac{Recall \cdot{Precision}}{Recall+Precision}}$$

這就是我們常聽到的`F1-score(F-measure)`

*Harmonic mean: <https://en.wikipedia.org/wiki/Harmonic_mean>*

- ROC and AUC   (*圖片以Undersampling為例*)

![](/assets/2019-10-08-Imbalanced-Data/roc.png)

ROC以及AUC對於Imbalanced data同樣是常用的evaluation method。尤其是在不同模型的情況下，我們要判別這個模型對於當前問題整體處理的效果是如何。
  - True Positive Rate(TPR, y-axis) = \\(TP/(TP+FN)\\)
  - False Positive Rate(FPR, x-axis) = \\(FP/(TN+FP)\\)

我們希望一個特定參數的模型他的 TPR 是好的，代表他夠找出大部分的 Positive samples。同時他的 FPR 是低的，也就是誤判Positive的機會也不大。對於一個模型的參數最理想的情況就是: `(TPR, FPR) = (1, 0)`。不過由於我們在比較不同模型，所以會把各個參數都列出來，並且繪製成曲線，這就是ROC。


然而我們有了適當的metrics去衡量我們的模型後，問題仍然沒有解決。因為樣本不平衡的情況依然存在，被標記為陽性(Labeled Positive)的樣本依然很少，同樣會造成演算法學習的問題。這樣樣本不平均的情況，目前通常利用重採樣(Resampling)的方式來解決。

## 重新採樣 Resampling
因為我們Data的數量相當不平衡，e.g, \\(\mid\mathcal{D_{majority}}\mid \gg \mid\mathcal{D_{minority}}\mid\\)，在這種情況下我們會想把他們的數量調整成一樣以便演算法學習，因此有兩種方式解決這個問題，而兩種處理方式各有優缺。
1. Oversampling: 產生數少量一方的data
2. Undersampling: 減少數量多那方的data



*[Q]對於特定algothim會不會遇到distribution不同的問題？*


**Oversampling**
- Random sampling

    Oversampling最直覺的方法就是利用原本少數類別的樣本去重複抽樣，藉此達到樣本數量平衡的情況。顯而易見，這種情況下效果不會太好，不僅如此，還會增加overfitting的機會。因為我們產生的是重複且原有的資料點，如果多次抽樣的點剛好是影響decision boundary最嚴重的點，這樣的情況就會更糟。

- SMOTE(**S**ynthetic **M**inority **O**versampling **T**echniqu**E**, 2002)

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

- Borderline-SMOTE(2005)
  
    這方法主要強調:對於二元分類來說，在邊界線(borderline)的樣本顯得更為重要，這些樣本對於模型訓練的影響是比較大的，所以只想要在邊界線附近利用SMOTE來產生synthetic樣本。在這個演算法中，最關鍵的就是怎麼去決定哪些是邊界樣本。假設所有的訓練資料集為： $\mathcal{T}$, 少數的資料集為 $\mathcal{P} = \\{p^{(1)}, p^{(2)}, p^{(3)}, ... , p^{(pnum)} \\}$，多數的資料為 $\mathcal{N} = \\{n^{(1)}, n^{(2)}, n^{(3)}, ... , n^{(nnum)}\\}$，如果是在borderline附近的資料集我們先記為 $\mathcal{Danger} = \\{\\}$。然後先利用 $\mathcal{P}$ 中的每個資料 $p^{(i)}$ 對 $\mathcal{T}$ 計算 m-nearest neighbors，然後在這些 neighbors 中，計算多數類別的樣本有多少個，數量記為 $m^{'}$ ($0 \leq m^{'}\leq m$)，並將情況分成三種：
    - $\small{m^{'} = m}$: 在 $p^{(i)}$ 附近的全是Majority class，代表 $p^{(i)}$是雜訊，將 $p^{(i)}$ 從 $\mathcal{P}$ 中移除。
    - $\small{0 <= m^{'} <= m/2}$: $p^{(i)}$ 附近都是相同類別的，所以應該不在邊界上。
    - $\small{m/2 <= m{'} <= m}$: 在 $p^{(i)}$ 附近有很多negative neighbors, 因此，這個 $p^{(i)}$ 值得參考，是在borderline，我們將這個樣本點加入到 $\mathcal{Danger}$。
    
    Borderline-SMOTE1: 然後對 $\mathcal{Danger}$ 中的每一個樣本對 $\mathcal{P}$ 計算KNN並進行SMOTE演算法，這就是第一種Borderline-SMOTE了。

    Borderline-SMOTE2: 因為目前在 $\mathcal{Danger}$ 的樣本點都是接近在邊線了，所以離邊線附近的negative points也很接近！那我們可不可以利用negative points來產生synthetic samples? Borderline-SMOTE2就是基於這樣的出發點設計出來的：利用 $\mathcal{N}$ 進行KNN並使用SMOTE產生synthetic samples。唯一的差別在於和 $\Delta$ 相乘的scalar $\eta$ 不再是從(0, 1)抽樣了，而是(0, 0.5)中抽樣，這樣會離邊線positive那一側較近。
    
    *我讀到這邊的時候，在想他是怎麼分配抽樣 $\mathcal{P}$ 和抽樣 $\mathcal{N}$的比例，有想過一種實作方式是我們直接對 $\mathcal{T}$做KNN，然後我們有會知道k samples的label, 單純從k samples抽樣的時候去選擇random scalar，像是 `upper_bound = 1 if sample is positive else 0.5` 然後再 `random(0, upper_bound)`。*

    至於，Borderline-SMOTE1和Borderline-SMOTE2哪一個比較好？在論文中對於不同的資料集是各有優缺，但是都比原本的SMOTE來的好就是了。

其餘的Oversampling方法還有`SVMSMOTE`, `ADASYN`, `Baysian Network/GAN`等，有機會再額外介紹。

**Undersampling**
  - The Condensed Nearest Neighbors Rule(1968)
  
    Condensed Nearest Neighbors是很早期的方法(雖然說後面的方法也沒有多近期)，背後核心的演算法是1-Nearest Neighbors。參考imblearn的文件，在imbalanced data的情境下，原本就是binary classification problem，所以我們可以初始化 \\(\mathcal{C} = \mathcal{D_{minority}}, ~\mathcal{S} = \mathcal{D_{majority}}\\)。為了要讓 \\(\mid\mathcal{S}\mid\\) 有機會利用1-NN來減少，首先會在 \\(\mathcal{S}\\) 中隨機挑一個樣本 \\(s^{(c)}\\) 加入到 \\(\mathcal{C}\\)，被挑選出來的樣本目前不會從 \\(\mathcal{S}\\) 中移除，先對 \\(\mathcal{C}\\) 訓練一個1-NN model。然後把 \\(\mathcal{S}\\) 的每一個樣本 \\(s^{(i)}\\) 利用1-NN model進行分類，比較預測值和他的標記Label，如果預測值和Label不同就把這個 \\(s^{(i)}\\) 從 \\(\mathcal{S}\\) 中移除，並加入到 \\(\mathcal{C}\\)。當遍歷整個 \\(\mathcal{S}\\) 的時候演算法就結束了。基於這樣逐漸擴大 \\(\mathcal{C}\\) 的方式會受到一開始隨機的 \\(s^{(c)}\\) 影響很大。 

    *這是1968年提出的方法了，當時也許不是拿來處理imbalanced data的問題，因為我自己是找不太到原始論文的內容，無法知道當初的出發點是什麼。CNN應該是比較廣泛的方法，只是剛好可以應用在Imbalanced data上面，不過他的計算量也確實很驚人。如果想知道實做方法可以看這邊[imblearn source code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/12b2e0d/imblearn/under_sampling/_prototype_selection/_condensed_nearest_neighbour.py#L152)直接看程式碼的註解就可以知道演算法架構，此外[Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#CNN_for_data_reduction)中有更詳細的介紹*
    
- Tomek Links(T-Links)

    為了解決CNN一開始挑選到不好的 \\(s^{(c)}\\) 的問題，Ian Tomek 想了一些方法來改善，並且Tomek Links是現在常見處理不平衡資料的方式。

- One-sided selection
- Informed Undersampling
- NearMiss

**The cross validation on undersampling**

## Algorithm Perspective
- Boosting: Adaboost -> 效果不好
- Xgboost?
**Ensemble method**

## Change loss function
- Weighting Strategy
- Cost-sensitive Learning Algorithm

## Anomaly detection
- 像是在偵測outliers一樣嗎？


最後，十分推薦imblearn這個package的網站，裡面可以看到許多examples: <https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/index.html>，裡面有很多處理imbalanced data的例子，從API也可以得知常用的metrics&algorithm。


## Questions
1. 為什麼Decision Tree, Random Forest不用擔心imbalance data issue?
2. 在Deep learning中，resampling依然奏效，可是deep learning需要的training data更為大量，有沒有什麼額外方法來解決這個問題？或是從模型設計的角度去思考?

## Related Research Papers
- [SMOTE: Synthetic Minority Over-sampling Technique, JAIR 2002](https://arxiv.org/pdf/1106.1813.pdf)
- [Borderline-SMOTE: A New Over-Sampling Method in
Imbalanced Data Sets Learning](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf)
- [Two Modifications of CNN, Ivan Tomek, 1976](https://www.semanticscholar.org/paper/Two-Modifications-of-CNN-Tomek/090a6772a1d69f07bfe7e89f99934294a0dac1b9)

## Reference:
- [大鼻David's Perspective: 不平衡資料的二元分類](https://taweihuang.hpd.io/2018/12/28/imbalanced-data-performance-metrics/)
- [Dealing with Imbalanced Data, Tara Boyle](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
- [知乎: 数据嗨客 第6期：不平衡数据处理](https://zhuanlan.zhihu.com/p/21406238)
- [Book: The Quest for Machine Learning: 100+ Interview Quesions for Algorithm Engineer](https://www.tenlong.com.tw/products/9787115487360)
- [Python package: imblearn](https://imbalanced-learn.readthedocs.io/en/stable/index.html)
- [Kaggle competition: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

others: latex size: http://www.sascha-frank.com/latex-font-size.html

