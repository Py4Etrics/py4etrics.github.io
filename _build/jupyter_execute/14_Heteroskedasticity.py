# 不均一分散

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import lmdiag
import wooldridge
from statsmodels.stats.api import het_breuschpagan, het_white
from seaborn import residplot
from statsmodels.stats.outliers_influence import reset_ramsey

## 説明

**＜仮定５（均一分散; Homogeneity）が満たされない場合＞**

仮定５の均一分散の下では説明変数は誤差項の分散に影響を与えない。即ち，

$$\text{Var}\left(u|x\right)=\sigma^2$$

この仮定に関連して次の点について留意する必要がある。
* 均一分散が満たされない場合でも，
    * 仮定１〜４のもとでOLS推定量 $\hat{\beta}_j$の不遍性と一致性は成立する。
    * $R^2$の解釈は変わらない。
* しかし，均一分散が満たされない場合，
    * OLS推定量の標準偏差の推定量である標準誤差$\text{se}\left(\hat{\beta}_j\right)$は無効となる。従って，$t$検定と$F$検定も無効になる。
    * 大標本特性（漸近的特性）も成立しない。従って，大標本であっても$t$検定と$F$検定も無効になる。

仮説を検証するということを目的とすると，検定が無効というのは致命的な問題である。特に，不均一分散（Heteroskedasticity）の問題は，横断面データを使うと頻繁に出てくる問題である。ではどのように対応すればよいのか。

---
**不均一分散頑健的推定（Heteroskedasticity-Robust Inference）**

この手法を使うと，OLS推定量の標準誤差が調整され**未知の**不均一分散であっても，$t$検定，$F$検
定が有効になるというものである。

（理由）均一分散であっても不均一分散であっても，$n\rightarrow\infty$の場合，不均一分散頑健的推定の$t$($F$)値は$t$($F$)分布に従う。言い換えると，標本の大きさが十分に大きければ，$t$($F$)値の分布は$t$($F$)分布で近似できるということである。

更なる利点は，通常のOLS推定の後に標準誤差の調整が施され，計算は`statsmodels`を使うと簡単におこなうことが可能である。

（注意）大標本でのみ有効。

---
不均一分散頑健的推定では，OLS推定の共分散行列（covariance matrix）と呼ばれる箇所を調整し，OLS推定量の標準誤差を修正する。その調整方法が複数提案されていおり，`statsmodels`では以下の種類に対応している。
* `HC0`: White (1980)の不均一分散頑健共分散行列推定
* `HC1`: MacKinnon and White (1985)の不均一分散頑健共分散行列推定v1
* `HC2`: MacKinnon and White (1985)の不均一分散頑健共分散行列推定v2
* `HC3`: MacKinnon and White (1985)の不均一分散頑健共分散行列推定v3

ここで`HC`は`H`eteroskedasticity-`C`onsistent Covariance Matrix Estimatorsの`H`と`C`。

不均一分散頑健共分散行列推定を使い計算した$t$値や$F$値を**頑健的$t$値**，**頑健的$F$値**と呼ぶことにする。

---
OLS推定量の不均一分散頑健標準偏差が簡単に計算できるのであれば，通常の標準偏差を使う必要はないのではないか，という疑問が生じる。この問を関して以下の点を考える必要がある。
* 通常の標準偏差を使う利点
    * 均一分散の場合（仮定1〜6（CLR仮定）），標本の大きさの大小に関わらず，$t$ ($F$)値の分布は**厳密に** $t$ ($F$)分布に従う。
* 不均一分散頑健標準偏差
    * 小標本の場合
        * 頑健的 $t$ ($F$)値の分布は必ずしも $t$ ($F$)分布に従うわけではない。その場合，$t$ ($F$)検定は無効となる。
    * 大標本の場合
        * $t$ ($F$)値の分布は $t$ ($F$)分布で**近似**され，$t$ ($F$)検定は有効である。
        * この結果は仮定１〜６（CLR仮定）のもとでも同じ。

従って，標本の大きさが「大標本」と判断できる場合（例えば，$n=1000$）以外は通常の標準偏差と不均一分散頑健標準偏差の両方を表示することを勧める。

## 頑健的$t$検定

`wooldridge`パッケージのデータセット`gpa3`を使い説明する。この例では大学のGPAと高校の成績や性別，人種などがどのような関係にあるかを探る。

gpa3 = wooldridge.data('gpa3').query('spring == 1')  # 春学期だけを抽出
wooldridge.data('gpa3', description=True)

`gpa2`に一部の変数の説明が続いている。

wooldridge.data('gpa2', description=True)

### OLS推定

被説明変数：
* `cumgpa`：累積GPA

説明変数
* `sat`：SATの成績
* `hsperc`：高校の成績の％点（上位から）
* `tothrs`：データ抽出時から学期までの時間？（`gpa3`の定義）
* `female`：女性ダミー変数（女性＝`1`）
* `black`：人種ダミー変数（黒人＝`1`）
* `white`：人種ダミー変数（白人＝`1`）

form_ols = 'cumgpa ~ sat + hsperc + tothrs + female + black + white'

mod_ols = ols(form_ols, data=gpa3)
res_ols = mod_ols.fit()

print(res_ols.summary().tables[1])

### 不均一分散頑健推定：方法１

上のOLSの結果を使い頑健$t$値を計算するために，`res_ols`のメソッド`.get_robustcov_results()`を使う。
* オプション`cov_type`は頑健性の計算法の指定（デフォルトは`CH1`）。
* オプション`use_t`は$t$検定を指定（デフォルトは`None`で「自動」に決められる）

res_robust = res_ols.get_robustcov_results(cov_type='HC3', use_t=True)

print(res_robust.summary().tables[1])

1. OLS推定量`coef`の値は同じであり，必ずそうなる。不均一分散頑健推定は，表の中で標準誤差，$t$値，$p$値，信頼区間に影響を与える。
2. 標準誤差`std err`を比べると，帰無仮説$\hat{\beta}_j=0$の棄却判断を覆すほど大きく変わる変数はない。これは不均一分散がそれほど大きな問題ではないことを示唆している。この点を確かめるために，`res_ols`の誤差項を図示してみる。

誤差項を図示する方法として２つを紹介する。一つ目は，`res_ols`の属性`.resid`を使う。`res_ols.resid`は`Pandas`の`Series`（シリーズ）なので，そのメソッド`plot()`を使い図示する。`style`はマーカーを指定するオプション。

res_ols.resid.plot(style='o')
pass

２つ目の方法は`plt.plot()`を使う。オップション`'o'`はマーカの指定である。

plt.plot(res_ols.resid, 'o')
pass

３つ目は`scatter()`を使う。`.index`は`res_ols.resd`の属性でインデックスを示す。

plt.scatter(res_ols.resid.index, res_ols.resid)
pass

### 不均一分散頑健推定：方法２

OLS推定をする際，`fit()`の関数に`.get_robustcov_results()`で使った同じオプションを追加すると頑健$t$値などを直接出力できる。

res_HC3 = ols(form_ols, data=gpa3).fit(cov_type='HC3', use_t=True)

print(res_HC3.summary().tables[1])

## 頑健的$F$検定

同じデータ`gpa3`を使い，黒人ダミーと白人ダミーの係数は両方とも`0`という仮説を検定する。

hypotheses = 'black = white = 0'

まず通常の$F$検定を考える。

f_test_ols = res_ols.f_test(hypotheses)

f_test_ols.summary()

返り値（左から）
* `F statistic`：$F$統計量
* `F p-value`：$F$の$p$値
* `df_denom`：分母の自由度
* `df_num`：分子の自由度

次に頑健$F$検定の方法を説明する。上の不均一分散頑健推定の方法２で使った`f_test_HC3`を使う。

f_test_HC3 = res_HC3.f_test(hypotheses)

f_test_HC3.summary()


$t$検定の場合と同じように，大きく変わる結果につながってはない。

## 均一分散の検定

均一分散の場合 $t$($F$)値は厳密に$t$($F$)分散に従う。それが故に，均一分散が好まれる理由である。ここでは均一分散の検定について考える。帰無仮説と対立仮説は以下となる。

$\text{H}_0$：誤差項は均一分散

$\text{H}_A$：帰無仮説は成立しない

２つの検定方法を考える。

### ブルーシュ・ペーガン（Breusch-Pagan）検定

データ`hprice1`を使って，住宅価格の決定要因を検討する。ここで考える均一分散の検定にBreusch-Pagan検定と呼ばれるもので，`statsmodels`のサブパッケージ`stats`の関数`het_breuschpagan`を使う。

hprice1 = wooldridge.data('hprice1')
wooldridge.data('hprice1', description=True)

以下で使う変数について。

被説明変数
* `price`：住宅価格（単位：1000ドル）

説明変数
* `lotsize`：土地面積（単位：平方フィート）
* `sqrft`：家の面積（単位：平方フィート）
* `bdrms`：寝室の数

#### 対数変換前

まず変数の変換をしない場合を考える。

form_h = 'price ~ lotsize + sqrft + bdrms'

res_h = ols(form_h, data=hprice1).fit()

print(res_h.summary().tables[1])

この結果に対してBreusch-Pagan検定をおこなう。`het_breuschpagan()`の引数について：
* 第１引き数：OLSの結果`res_h`の属性`.resid`で誤差項の値
* 第２引き数：OLSの結果`res_h`の属性`.model`の属性`exog`を使い定数項を含む説明変数の値

het_breuschpagan(res_h.resid, res_h.model.exog)

返り値（上から）
* `LM statistic`：$LM$統計量
* `LM p-value`：$LM$の$p$値
* `F statistics`：$F$統計量
* `F p-value`：$F$の$p$値

$LM$検定とはLagrange Multiplier検定のことで，大標本の場合に仮定１〜４（GM仮説）のもとで成立する。一般にBreusch-Pagan検定は$LM$統計量を使ったものを指すが，$F$統計量としても計算できる。

5%の有意水準で帰無仮説（$\text{H}_0$：誤差項は均一分散）を棄却でき，不均一分散の可能性が高い。対処法として変数の変換が挙げられる。

#### 対数変換

`bdrms`以外を対数変換する。

form_h_log = 'np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms'

res_h_log = ols(form_h_log, data=hprice1).fit()

print(res_h_log.summary().tables[1])

het_breuschpagan(res_h_log.resid, res_h_log.model.exog)

5%の有意水準でお帰無仮説を棄却できない。即ち，対立仮説（$\text{H}_A$：帰無仮説は成立しない）を採択し，均一分散の可能性が高い。

### ホワイト（White）検定

この検定はOLS推定量の標準誤差と検定統計量を無効にする不均一分散を主な対象としており，Breusch-Pagan検定よりもより一般的な式に基づいて検定をおこなう。`statsmodels`のサブパッケージ`stats`の関数`het_breuschpagan`を使う。

`hprice`のデータを使った上の例を使う。

`het_white()`の引数：
* 第１引き数：OLSの結果`res_h`の属性`.resid`で誤差項の値
* 第２引き数：OLSの結果`res_h`の属性`.model`の属性`exog`を使い定数項を含む説明変数の値

#### 対数変換前

het_white(res_h.resid, res_h.model.exog)

返り値（上から）
* `LM statistic`：$LM$統計量
* `LM p-value`：$LM$の$p$値
* `F statistics`：$F$統計量
* `F p-value`：$F$の$p$値

一般にWhite検定は$LM$統計量を使ったものを指すが，$F$統計量としても計算できる。

5%の有意水準で帰無仮説（$\text{H}_0$：誤差項は均一分散）を棄却でき，不均一分散の可能性が高い。対処法として変数の変換が挙げられる。

#### 対数変換後

het_white(res_h_log.resid, res_h_log.model.exog)

5%の有意水準で帰無仮説を棄却できない。即ち，対立仮説（$\text{H}_A$：帰無仮説は成立しない）を採択し，均一分散の可能性が高い。


## 残差：図示と線形性

## 図示

仮定４〜６は残差に関するものであり，残差をプロットし不均一分散や非線形性を確認することは回帰分析の重要なステップである。

残差を図示する方法として`lmdiag`以外に以下を紹介する。
1. `matplotlib`を直接使う
2. `seaborn`というパッケージの中にある関数`residplot`を使う

上で計算した`res_h`と`res_h_log`を利用し
* 横軸：被説明変数の予測値（メソッド`.fittedvalues`）
* 縦軸：残差（メソッド`.resid`）

となる図を作成する。

### `lmdiag`

対数変換前

plt.figure(figsize=(8,7))
lmdiag.plot(res_h)
pass

対数変換後

plt.figure(figsize=(8,7))
lmdiag.plot(res_h_log)
pass

対数変換により残差の変化がより均一的になり，Residuals vs. LeverageのCook's Distanceを見ても外れ値がなくなっている。

### `Matplotlib`の`plot()`

対数変換前

plt.scatter(res_h.fittedvalues, res_h.resid)
pass

対数変換後

plt.scatter(res_h_log.fittedvalues, res_h_log.resid)
pass

対数変換により残差の変化がより均一的になったのが確認できる。

### `seaborn`の`residplot()`

`seaborn`は`matplotlib`を利用し様々な図を描ける。`seaborn`については[このサイト](https://seaborn.pydata.org/index.html)を参照。

通常`import seaborn as sns`でインポートすることが多いようであるが，ここでは`residplot`のみをインポートしている。

`residplot()`は散布図を作成する関数である。
* 第１引き数：横軸の変数
    * 被説明変数を設定することを勧める。
* 第２引き数：縦軸の変数
    * 残差
* オプション
    * `lowerss=True`（デフォルトは`False`）にすると，散布図にベスト・フィットする**曲線**を表示する。

---
対数変換前

residplot(res_h.fittedvalues, res_h.resid, lowess=True)
pass

対数変換後

residplot(res_h_log.fittedvalues, res_h_log.resid, lowess=True)
pass

## 線形性

仮定１で回帰式は線形となっているが，その仮定が正しければ，上の残差の散布図は概ね平行にそして０を中心に上下等間隔に散らばることになる。そのパターンに比較的に近いのは対数変換**後**の図である。不均一分散の場合は，そのパターンが大きく崩れている場合であり，その原因のその１つに回帰式の特定化の間違いがある。例えば，説明変数の２乗が含まれるべきなのに欠落している場合が挙げられる。極端な場合，残差の散布図は$U$字または逆$U$字のような形になりえる。

一方，線形性の検定もある。以下ではその１つである RESET (Regression Specification Error Test) 検定を考える。

この検定の考え方はそれほど難しくはない。次の回帰式を推定するとしよう。

$$y=\beta_0+\beta_1x_1+\beta_2x_2+u$$

この線形式が正しければ，$x_1^2$や$x_2^3$等を追加しても統計的に有意ではないはずである。さらに，この線形回帰式の予測値$\hat{y}$は$x_1$や$x_2$の線形になっているため、$\hat{y}^2$や$\hat{y}^3$は$x_1$や$x_2$の非線形となっている。従って，次式を推計し，もし非線形の効果がなければ$\delta_1$も$\delta_2$も有意ではないはずである。

$$y=\beta_0+\beta_1x_1+\beta_2x_2+\delta_1\hat{y}^2+\delta_2\hat{y}^3$$

この考えに基づいて以下の仮説を検定する。

$\text{H}_0:\;\delta_1=\delta_2=0$（線形回帰式が正しい）

$\text{H}_A$: $\text{H}_0$は成立しない 

＜コメント＞
* 通常は$y$の3乗まで含めれば十分であろう。
* 大標本のもとで$F$検定統計値は$F$分布に従う。


---
`statsmodels`のサブパッケージ`.stats.outliers_influence`のRESET検定用の関数`reset_ramsey`を使う。（`ramsey`はこの検定を考案した学者名前）

`reset_ramsey()`の使い方：
* 引き数：OLS推定の結果
* オプションの`degree`（デフォルトは5）は$y$の何乗までを含めるかを指定する。

**対数変換前**

reset_ramsey(res_h,degree=3).summary()

返り値
* `F`: $F$統計量
* `p`: $p$値
* `df_denom`: 分母の自由度
* `df_num`: 分子の自由度（２乗以上の制約数）

**対数変換後**

reset_ramsey(res_h_log,degree=3)

5%の有意水準のもとで，`res_h`のの帰無仮説を棄却できるが，`res_h_log`では棄却できない。後者の推計式がより適しているようである。