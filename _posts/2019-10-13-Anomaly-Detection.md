---
layout: post
title: Coursera Maching Learning, Andrew Ng - Anomaly Detection 異常偵測讀書筆記
maxjax: true
---

之前一直沒有好好拜讀Andrew Ng在Coursera開的Machine Learning課程。七月開始接觸到Andrew Ng開的Deep Learning專項課程後，覺得實在是太優良了，所以不得不好好學習一下。正好，最近接觸到imbalanced data的議題，查了一下想知道imbalanced data可不可以利用anomaly detection的角度去看待，正好發現Andrew Ng有開相關的內容。

先簡單說一下區別：Imbalanced data一般是在處理supervised learning的問題，在Andrew Ng的課程中，雖然我們有data label，但是不是採用supervised learning的方式去學習。而是利用資料建立Gaussian Model並且設定threshold $\varepsilon$來決定該樣本是不是anomalous data，後面我們有更詳細的介紹。

以下的筆記對應到課程投影片:

## Problem Motivation
![](/assets/2019-10-13-Anomaly-Detection/pv-1.png)
假設我們要判別aircraft的異常偵測，這裡資料只有兩個features $x_1$, $x_2$ 為例, 綠色的樣本實際上是異常的資料。我們可以假設異常的樣本發生的機會很小，因此利用Gaussian Distribution去對現有正常(Normal)的資料建模。

下圖等高線代表資料出現的`Probability Density Function(p.d.f)`。如果 $x_1$, $x_2$ 在等高線同一圈代表$p(x_1) = p(x_2)$。然後我們可以利用設定一個threshold $\varepsilon$ 來決定樣本是不是異常。

![](/assets/2019-10-13-Anomaly-Detection/pv-2.png)
 
Anomaly detection常用在:
- Fraud detection
- Manufacturing
- Monitoring computers in a data center.

## Gaussian Distribution & Parameter Estimation
這部份Andrew Ng是在介紹如何用Gaussian Distribution去估計資料真實的分佈狀況。下面是Gaussian Distribution的公式:

$$
p(x;\mu, {\sigma}^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp({-\frac{(x-\mu)^2}{2{\sigma}^2}}),
\\
x \sim N(\mu, \sigma^2), \ x \in \mathbb{R}
$$

估計 $\mu$, ${\sigma}^2$ 的方式如下:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \ {\sigma}^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2
$$

唯一要注意的是，雖然在統計上，我們要用樣本估計母體的參數時，$\frac{1}{m}$ 應該要是 $\frac{1}{m-1}$。但是，Andrew Ng說：在Machine Learning上因為資料量 $\mid D \mid$ 會很大，影響不大，所以可以用原本的公式去估計參數 $\mu$, $\sigma$。

## Algorithm
我們有了基礎的想法後，要怎麼利用Gaussian Distribution來做異常偵測？

假設：

$$
\text{Training Set: } \{ x^{(1)}, ... , x^{(m)}\}
\\
\text{Each example is: } x \in \mathbb{R^n}
\\
\text{Each feature d in example is: } x_d
$$

然後我們可以對資料的每一維度去估計他的 $\mu_d$, $\sigma_d$，然後每筆資料的第d維數據可以以此抽樣：$x_d \sim N(\mu_d, {\sigma_d}^2)$，我們假設每個features彼此之間不相關，則normalized p.d.f可以以下面方式估計:

$$
p(x) = p(x_1;\mu_1, \sigma_1^2)\  p(x_2;\mu_2, \sigma_2^2) \ p(x_3;\mu_3, \sigma_3^2) \ ... 
\ p(x_n;\mu_n, \sigma_n^2)
\\
\ \ = \prod_{j=1}^{n} p(x_j;\mu_j, {\sigma_j}^2)
$$

完整的演算法則是:

1.Choose features $x_i$ that you think might be indicative of anomalous examples.

2.Fit parameters $\mu_1, ..., \mu_n, \sigma_1^2, ..., \sigma_n^2$
 
$$
\\
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}_{j}
\\
{\sigma_j}^2 = \frac{1}{m} \sum_{i=1}^m (x^{(i)}_j - \mu_j)^2
$$

3.Given new example x, compute $p(x)$:

$$
p(x) = \prod_{j=1}^{n} p(x_j;\mu_j, {\sigma_j}^2) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_j} \exp(-\frac{(x_j - \mu_j)^2}{2 \sigma_j})
$$

4.if $p(x) \lt \varepsilon$,  $x$ is anomolous. $\varepsilon$ is the threshold condition on the probability density.

所以如果有 n 維的資料，我們在這演算法中會有 `n + n = 2n` 個參數。然後我們必須不斷測試threshold: $\varepsilon$ 是不是適合的。雖然這建立在各個features都是`independent`的況下，但是Andrew Ng說在實務上還是可用的。

![](/assets/2019-10-13-Anomaly-Detection/alg-1.png)

## Developing and Evaluating an Anomaly Detection System

在評估這個方法的時候，資料集會分成三種:
1. Training Set: 訓練用演算法用
2. Cross Validation Set: 調整演算法參數用 ($\varepsilon$)
3. Test Set: 最後挑選出一個特定參數的演算法，評估它的效能用

我們通常會用`正常normal (y=0)`的資料作為Training Set，實際上因為是用Gaussian Distribution去估算資料發生的機率，所以不需要考慮他的Label是什麼，因為這樣Andrew Ng才會在課程中說這是`unlabeled data`，但是他也有提到裡面摻雜一些`anomalous data(y=1)`也是沒關係的。另外，Andrew Ng有特別提到：Cross Validation Set和Test Set不能是同樣的資料集。

以Aircraft engines的異常偵測為例，假設在我們資料中有10000筆標記是正常的, 只有20筆標記為不正常 (anomalous < 0.2%)。可以將資料集分成:
![](/assets/2019-10-13-Anomaly-Detection/eval-1.png)

藍色的部份是正確的，紫色的部份是錯誤的。其中，紫色的例子在anomalous中有typo:`10 anomalous -> 20 anomalous`這點在Coursera的勘誤表有提到，另外另外代表Cross Validation Set和Test Set是一樣的。

在Cross Validation Set的時候，我們要評估演算法結果的好壞，Andrew Ng提供以下幾種方式:
- True Positive, false positive, false negative, true negative
- Precision/Recall
- F1-score

以上大概就是做Anomaly Detection的整個流程了。

## Anomaly Detection vs. Supervised Learning

這段是我覺得很重要的章節，這個章節比較了Anomaly Detection和Supervised Learining的不同，以及使用時機。Andrew Ng在這裡提到了一個觀念是我覺得特別需要留意的：**Anomaly Detection和Supervised Learning最大的不同是我們能不能藉由現有的資料，知道未來anomalous資料可能是怎麼樣？**。

像是在檢測機器異常時，我們不能知道所有異常的種類是如何，所以我們會採取Anomalous Detection的方式來處理問題。但是如果是在Spam Eamil Detection的問題中，因為這個應用發展很成熟了，我們也大概知道Spam的類型會包含什麼特徵以及因素，所以會直接採取Supervised Learning的方式。

也是因為如此，我們再做Anomaly Detection的時候，是對Negative/Normal data去建模而不是對Postive/Anomalous Data建模(太少了，不足以提供足夠資訊)。

![](/assets/2019-10-13-Anomaly-Detection/anomaly_vs_sl-1.png)

這裡Fraud Detection的藍色箭頭指向右邊代表：如果有足夠的Positive data的話，我們就可以利用Supervised Learning來處理。

![](/assets/2019-10-13-Anomaly-Detection/anomaly_vs_sl-2.png)

## Choosing what features to use

Andrew Ng有提到features的選擇對於Anomaly Detection的結果影響很大。其一是對於Non-Gaussian features的轉換，其二是如何做Error Analysis。

### Non-Gaussian features

我們的Anomaly Detection是建立在各個features也是利用Gaussian Distribution來估計取得的，所以我們希望每個features都能夠滿足Gaussian的條件。所以Andrew Ng這邊提供了一些features transformation的方法讓原本的features分佈更接近Gaussian Distribution。

下面圖表的x-axis: $x$ 其實代表的是其中一個feature $x_d$。
1. $log(x_d + c)$, e.g, $log(x_{j} + 1)$
2. $x_d^{(1/c)}$, e.g, $x_{j}^{\prime} = x_{j}^{0.05}$
![](/assets/2019-10-13-Anomaly-Detection/features-1.png)

### Error analysis for anomaly detection
理想情況是我們希望 $p(x^{(normal)}) \gt p(x^{(anomalous)})$，但是這件事情不可能總是發生，會有`異常樣本的p.d.f > 正常樣本的p.d.f`的情況。這時候，要就把那些樣本找出來進一步分析原因。並且回到Feature Engineering的角度去想：是不是能產生新的適合的feature？

下面Andrew Ng舉了資料中心監測電腦的情況作為例子，我們在選擇features的時候，會考量一項資料是不是會不正常的很大或是很小的數值。

假設資料中心的每一台電腦的CPU load和Network traffic應該是正相關的。當一台電腦產生infinite loop的異常時，CPU load會很高，但是network traffic卻依然沒有成長。但是在我們原有的features卻不能顯現這件事情，所以必須額外產生新的feature: $x_5$。 $x_6$ 則是和 $x_5$ 想表達的觀念相同，但是數值變化上更為劇烈。
![](/assets/2019-10-13-Anomaly-Detection/features-2.png)

## Multivariate Gaussian Distribution

### Motivating example: Monitoring machines in a data center
最後一個章節是Optional的，不過我覺得十分值得學習。前面Anomaly Detection方法建立在**個別的特徵彼此獨立，各個不相關**的前提上。所以我們必須像上面一樣去產生 $x_5$ 才能夠知道 $x_3$, $x_4$ 之間的關係。並且，他們也不見得真的是獨立的，所以我們總是會想: **有沒有一個辦法能不能衡量出這些features之間的關係，又同時具備Gaussian的性質？** Multivariate Gaussian 就是在做這樣的事情！

![](/assets/2019-10-13-Anomaly-Detection/multi-gaussian-1.png)

在上面的例子，可以看到 $x_1$, $x_2$ 是正相關的。所以我們得到的p.d.f等高線應該要是右上-左下的橢圓會比較合理一點。但是在原本的方法中，我們不能做到這件事情，而會將左圖中左上的`藍色X點`和其他的藍色X點誤判成相同可能性。但是Multivariate Gaussian可以處理這樣的問題。

### Multivariate Gaussian (Normal) distribution

我們這次直接對整個樣本點 $x$ 去建模，而不是個別的features。($x \in \mathbb{R}^n$)。

Parameters: $\mu \in \mathbb{R}^n$, $\Sigma \in \mathbb{R}^{n \times n}$ (covariance matrix)

$$
p(x;\mu, \Sigma) = \frac{1}{(2\pi)^{n/2} {\mid\Sigma\mid}^{1/2}} \ \exp ({-\frac{1}{2}}(x-\mu)^{T} \ \Sigma^{-1} (x-\mu)),
\\
\mid\Sigma\mid = \text{determinant of} \ \Sigma, \det(\Sigma)
$$

\
以下是一些Andrew Ng用來說明Multivariate Gaussian的例子， 可以留意一下$\Sigma_{ij} = \Sigma_{ji}$，並且原本的模型不能衡量 $\Sigma_{ij}$ 的關係，Covariance Matrix的上下三角都是 $0$，只有對角線有值，從視覺化來看會是一個圓體。

![](/assets/2019-10-13-Anomaly-Detection/multi-gaussian-2.png)

- $\Sigma_{ij}$ is positive:
![](/assets/2019-10-13-Anomaly-Detection/multi-gaussian-3.png)

- $\Sigma_{ij}$ is negative:
![](/assets/2019-10-13-Anomaly-Detection/multi-gaussian-4.png)

### Algorithm

1.Fit model $p(x)$ by setting:

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}
\\
\Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^T
$$

2.Given a new example $x$, compute:

$$
p(x;\mu, \Sigma) = \frac{1}{(2\pi)^{n/2} {\mid\Sigma\mid}^{1/2}} \ \exp ({-\frac{1}{2}}(x-\mu)^{T} \ \Sigma^{-1} (x-\mu))
$$

3.Flag an anomaly if $p(x) \lt \varepsilon$

### Origin Model vs. Multivariate Gaussian

![](/assets/2019-10-13-Anomaly-Detection/multi-gaussian-5.png)

最後，Andrew Ng比較了兩個模型不同的地方。Multivariate Gaussian在使用的時候，比較有一些限制條件。

Origin Model: 如果要捕捉到features之間的關係，必須要自己創造新的feature。但是訓練所需的樣本較少，`Training Set size = m, m = 50 or 100`都可以拿來訓練。
- \# of parameters: $2n$

Multivariate Model: 因為要計算 $\Sigma^{-1}$ 的關係，需要確保：$\Sigma$ is invertible，所以樣本的數量必須比特徵的數量還要多 `m > n`，Andrew Ng自己個人則是說 `m >= 10n` 時才會使用。在上圖左側，Computationally more expensive的下方有寫一個Matrix，那個意思是在說：我們使用Multivariate Gaussian的時候要注意Inveritble的問題。如果，原先就有 $x_1 = x_2, \ x_3 = x_4 + x_5$ 這樣`線性相依(Linear Dependent)`的關係，就會造成 $\Sigma$ 是不可逆的。所以使用Multivariate Gaussian的時候，Andrew Ng建議: 
1. Check m >= n
2. Check for redundant features.

- \# of parameters: $\frac{n^2}{2} + n$, 因為 $\Sigma_{ij} = \Sigma_{ji}$

## 結語
這個課程雖然只有一小時，可是裡面涵蓋的內容十分有架構，Andrew Ng說明的也很清楚，把一般會想到的疑問: Anomaly Detection vs. Supervised Learning, Features Engineering, Original Gaussian Model vs. Multivariate Gaussian都涵蓋了。謝謝 Andrew Ng老師在Coursera上開課。