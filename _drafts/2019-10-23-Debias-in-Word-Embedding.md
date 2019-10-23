---
layout: post
title: Debias in Word Embedding
mathjax: true
---

## 前言

這個筆記取材自 `Deeplearning.ai` series course: Word Vector Representation。

在處理word embedding的過程中，我們常常利用word embedding $e_i - e_j$ 的方式計算出他們的差距。因為，embedding可以想成是Machine Learning從大量語料庫(corpus)學習知識後再降維到一個空間的結果。在embedding space中的距離往往可以顯現不同字(word)之間的差距，並且可以利用 $e_i - e_j$ == $e_k - e_l$的方式去判別embedding pair的相似程度。

但是，我們也可以從word embedding觀察到一些現象：性別偏見，種族偏見。

## 發現問題

不囉唆，直接從作業中看例子:
```python
print ('List of names and their similarities with constructed vector:')

# define the gender direction vector
g = word_to_vec_map['woman'] - word_to_vec_map['man']

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle',
            'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
```
Result:
```
List of names and their similarities with constructed vector:

john -0.23163356146
marie 0.315597935396
sophie 0.318687898594
ronaldo -0.312447968503
priya 0.17632041839
rahul -0.169154710392
danielle 0.243932992163
reza -0.079304296722
katy 0.283106865957
yasmin 0.233138577679
```
從上面的例子看出來，g 是在衡量 $e_{woman}$ 跟 $e_{man}$ 的差距，而這個差距會反應成跟 gender 相關的 vector。然後我們利用這個和 vector 和英文人名做比較。發現：女生的人名會計算出 positve 的結果，而男生的名子會是 negative，**代表特定人名與性別是有相關的**。

如果，我們今天去比較的對象不是人名，而是職業或是其他的字詞，結果又是如何？
```python
print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior',
             'doctor', 'tree', 'receptionist','technology', 'fashion',
             'teacher', 'engineer', 'pilot','computer', 'singer']

for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
```
Result:
```
Other words and their similarities:

lipstick 0.276919162564
guns -0.18884855679
science -0.0608290654093
arts 0.00818931238588
literature 0.0647250443346
warrior -0.209201646411
doctor 0.118952894109
tree -0.0708939917548
receptionist 0.330779417506
technology -0.131937324476
fashion 0.0356389462577
teacher 0.179209234318
engineer -0.0803928049452
pilot 0.00107644989919
computer -0.103303588739
singer 0.185005181365
```

結果顯示：**在`receptionist`, `singer`的部份明顯偏向女性，在`computer`則偏向男性。**但是我們不希望一開始建立model的時候，model就帶有這樣的偏見。(這裡的corpus是利用Wikipedia訓練的)後續，Andrew Ng 就介紹了一個方法來 debias 計算的結果。

## 如何處理Bias問題?

我們利用Bolukbasi等人的方法，想法是: 有些字詞之間的距離必須反應性別關係，像是`actor/actress`, `grandfather/grandmother`要讓他們保有和`gender`相關的向量差距。但是針對`receptionist`這類的字詞應該消除`gender`相關的向量。

假設我們的word embedding的dimension: (50, 1) 我們可以想成這50個維度做線性組合後，會分成和性別相關的`bias-axis`，以及不相關的`orthogonal-axis`，在數學上`orthogonal`代表垂直，沒有相關性。所以我們同樣可以將每一個字詞的word embedding $e_{word}$ 投影分成兩個分量: $e_{word}^{biased}$ ,  $e_{word}^{debiased}$。為了對應到下面的例子，我們將`bias-axis`的方向向量定義為 $g$, 另一個則是 $g_{\perp}$。

對不同屬性的字詞，我們將用不同方法來處理 bias：
- Neutralization: 讓原本有偏見的詞變成沒有偏見
- Equalization：保有原本偏見，但是將差異縮小化

### Neutralization
對於一些跟性別無關的字，我們希望它回歸中性，這個操作過程叫做neutralization(中和)。因此我們要先找出bias的部份，再加以移除它。

$$
e^{bias\_component} = \frac{e \cdot g}{||g||_{2}^{2}} * g
\\
e^{debias\_component} = e - e^{bias\_component}
$$

![](/assets/2019-10-24-Debias-in-Word-Embedding/neutral.png)

```python
def neutralize(word, g, word_to_vec_map):
    
    # Select word vector representation of "word". Use word_to_vec_map 
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent using the formula give above.
    e_biascomponent = (np.dot(e, g) / np.square(np.linalg.norm(g)))*g
      
    # Neutralize e by substracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e - e_biascomponent
    
    return e_debiased
```

我們比較一下有無經過 neutralization 時，該 word embedding e 在 bias diretion(g) 的分量是多少
```python
e = "receptionist"
print("before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("after neutralizing: ", cosine_similarity(e_debiased, g))
```
Result:
```
before neutralizing:  0.330779417506
after neutralizing:  -5.84103233224e-18
```
後者是因為數值計算的關係而導致有一個極小值，但是會很接近0，代表bias被消除。

### Equalization
針對另一類的詞像是：`actor/actress` pair，他們在意義上病沒有什麼不同，單純只有 gender 的部份不同而造成向量上的差距。所以，我們希望改造一下原本這類的 word embedding 讓他們投影到 $g_{\perp}$ 的量是相同的，而 $g_{bias}$ 也相同。

很直覺的調整方式就是：
1. 在 $g_{\perp}$ 的方向讓他們保持有相同的向量。
2. 在 $g$ 的方向讓他們能夠展現一些原有的性質。

![](/assets/2019-10-24-Debias-in-Word-Embedding/equalize10.png)

在 Andrew Ng 的課程中，調整方式如下：
$$ \mu = \frac{e_{w1} + e_{w2}}{2}\tag{1}$$ 

$$ \mu_{B} = \frac {\mu \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{2}$$ 

$$\mu_{\perp} = \mu - \mu_{B} \tag{3}$$

$$ e_{w1B} = \frac {e_{w1} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{4}$$ 
$$ e_{w2B} = \frac {e_{w2} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{5}$$

$$e_{w1B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w1B}} - \mu_B} {||(e_{w1} - \mu_{\perp}) - \mu_B||} \tag{6}$$

$$e_{w2B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w2B}} - \mu_B} {||(e_{w2} - \mu_{\perp}) - \mu_B||} \tag{7}$$

$$e_1 = e_{w1B}^{corrected} + \mu_{\perp} \tag{8}$$

$$e_2 = e_{w2B}^{corrected} + \mu_{\perp} \tag{9}$$


看起來需要計算很多東西，但是實際上就是把原本的 $e_{w1}$, $e_{w2}$ 利用 $\mu_{\perp}$ 以及 $\mu_{B}$ 來組合。因為在 $g_{\perp}$ 方向要讓他們一樣所以直接取 $\mu_{\perp}$ 作為分量，見(8), (9)式。

至於，$g$ 的方向則是取各自原本的分量 $e_{w1B}$, $e_{w2B}$ 與 $\mu_{B}$ 相減做調整，並且調整一下比例(分母)。前項的 $\sqrt{...}$ 也是在調整比例，當 $\mu_{\perp}$ 越大的時候，投影在`bias-axis`上的分量也相對縮小。

論文上的版本更簡潔一些：

$$\mu = \sum_{w \in E} w/|E| \tag{10}$$

$$\nu = \mu - \mu_B \tag{11}$$

$$\text{For each} \ w \in E, \ \text{new embedding:} \ \vec{w} = \nu + \sqrt{1 - {||\nu||}^{2}} \  \frac{\vec{w_B} - \mu_B}{||\vec{w_B} - \mu_B||} \tag{12}$$

上面的 $\nu$ 即是 $\mu_{\perp}$。值得注意的是分母 $\vec{w_B}$ 和 $e_{wi} - \mu_{\perp}$ 會是不一樣的，會影響到 $\vec{w_B} - \mu_{B}$ 分量的大小。至於實際上會有什麼效果？我自己是認為他們效果是一樣的，至少都是在將 `bias-axis` 方向的向量調整到一樣。在很極端很極端的狀況下，也就是在 $w_1$, $w_2$ 在 $g_{\perp}$ 差距很大的情況下，兩者的數值會有差距。但是單就比例來說是相同的，所以調整後依舊滿足 $w_{1B} = w_{2B} (e_{w1B} = e_{w2B})$。

以上大概就是 Debias Word Embedding 的方法。能夠"發現這個問題"且進一步改進效能很令人敬佩。

##  Reference:
- [Paper: Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)

- Jupyter note from Word Vector Representation "Operations on word vectors - v2", deeplearning.ai
  
- Image Credits: deeplearning.ai