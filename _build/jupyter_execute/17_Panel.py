# パネル・データ分析

import numpy as np
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
from linearmodels.panel.data import PanelData
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, compare
from collections import OrderedDict
import wooldridge
from statsmodels.formula.api import ols

パネル・データを使った次のモデルについて説明する。
* 固定効果モデル（Fixed Effects Model）
* ランダム効果モデル（Random Effects Model）
* 相関ランダム効果モデル（Correlated Random Effects Model） 

## 固定効果モデル

### 説明

パネル・データを使う場合の問題は観察単位の異質性を捉える変数 $a_i$ が説明変数 $x_{it}$ と相関することにより，単純なOLS推定量に異質性バイアスが発生することである。その対処方法として，回帰式から$a_i$をなくす１階差分推定について説明した。この章では，同じように$a_i$をなくす代替推定法として固定効果推定（Fixed Effects Estimator）について考える。まず，その基礎となる固定効果モデル（Fixed Effects Model; FEモデル）について解説する。

次式を考えよう。

$$y_{it}= \beta_0 + a_i + \beta_1x_{it}+u_{it}\qquad i=1,2,..,n\quad t=0,1,...,T\qquad\qquad\left(\text{式１}\right)$$

両辺のそれぞれの変数の時間に対しての平均を計算すると次式となる。

$$\bar{y}_i= \beta_0 + a_i + \beta_1\bar{x}_i+\bar{u}_i\qquad\qquad\left(\text{式２}\right)$$

（注意）$a_i$は時間に対して不変なため，$a_i$はそのままの値を取る。

(式１)と(式２)の差を取ると次の固定効果推定式が導出できる。

$$\ddot{y}_i = \beta_1\ddot{x}_i+\ddot{u}_i\qquad\qquad\left(\text{式３}\right)$$

ここで
* $\ddot{y}_i=y_i-\bar{y}$
* $\ddot{x}_i=x_i-\bar{x}$
* $\ddot{u}_i=u_i-\bar{u}$

は平均からの乖離（demeaned values）。

＜良い点＞
* （式３）には $a_i$ がない。
* $a_i$ と $x_{it}$, $t=1,2,..,T$ を所与として $u_{it}$ の平均は0の仮定の下で，OLS推定量 $\hat{\beta}_1$ は
    * 不偏性を満たす。
    * $T$を一定として標本の大きさが十分に大きい場合，一致性を満たす。
    * 固定効果推定量（Fixed Effects Estimator or FE Estimator）もしくは Between Estimatorと呼ばれる。
    
＜悪い点＞
* 時間に対して一定の説明変数は，$a_i$と同じように，（式３）に残らない。従って，時間不変の変数の効果を推定することはできない。

### 「手計算」による推定

`wagepan`というデータセットを使い，賃金に対する労働組合などの変数の効果を推定する。

wagepan = wooldridge.data('wagepan')
wooldridge.data('wagepan', description=True)

被説明変数
* `lwage`：賃金（対数）

説明変数
* `union`：労働組合参加（ダミー変数）
* `married`：未・既婚（ダミー変数）
* `exper`：労働市場参加年数
*  `d81`：1981年のダミー変数
*  `d82`：1982年のダミー変数
*  `d83`：1983年のダミー変数
*  `d84`：1984年のダミー変数
*  `d85`：1985年のダミー変数
*  `d86`：1986年のダミー変数
*  `d87`：1987年のダミー変数

`DataFrame`の`groupby`を使って変数をグループ化する際に使う変数
* `nr`：労働者のID

```{admonition} コメント
:class: tip
時間に対して変化しない変数使えない。例えば，
* `educ`（説明変数に入れば教育の収益率が推定可能である。）
* `black`，`hisp`（を使うと人種間の賃金格差も推定できる。）

は固定効果モデル回帰式には入れることができない。
```

# 説明変数のリスト
exog = ['married','union','expersq','d81','d82','d83','d84','d85','d86','d87']

# 全ての変数のリスト
var = ['lwage']+exog

# 使う変数だけで構成されるDataFrame
df = wagepan.loc[:,['nr']+var]

# varの平均からの乖離を計算（下の説明（１）を参照）
df_g = df.groupby('nr')
df_mean = df_g[var].transform('mean')
df_md = df.loc[:,var]-df_mean

# 説明変数の行列（下の説明（２）を参照）
X = df_md.loc[:,exog].values

# 被説明変数のベクトル
Y = df_md.loc[:,'lwage'].values

# OLSの計算
params = np.linalg.inv((X.T)@X)@(X.T)@Y

# 結果の表示（下の説明（３）を参照）
for idx, name in enumerate(exog):
    print(f'{name}: {params[idx]:.4}')

説明（１）
* `df.groupby('nr')`：`nr`でグループ化したオブジェクトを作成する（番外編のGapminderを参照）。
* `df_g[var].transform('mean')`
    * `df_g[var]`：グループ化計算のために`var`の変数だけを使うことを指定する。
    * `.transform('mean')`：指定された変数（`var`）のグループ内平均（それぞれの`nr`内での平均）を計算し，その平均で構成される`DataFrame`を作成する。作成された`DataFrame`の行数は`df`と同じになり，グループ内平均が同じ列内でリピートされることになる。
* `df.loc[:,var]-df_mean`：`var`のそれぞれの変数の平均からの乖離を計算する。

説明（２）
* `.values`は`DataFrame`を`numpy`の`array`として返す。

説明（３）
* `enumerate(exog)`は`exog`の変数のインデックスと変数名の両方をタプルとして返す。
    * `for`ループで`enumerate()`は，引数にリストや`array`を入れると2つの値を返す。
    * 1つ目の返り値：要素のインデックス
    * 2つ目の返り値：要素自体
    * 例えば，`enumerate(['A','B','C'])`であれば，返り値は`(0,'A')`，`(1,'B')`，`(2,'C')`となる。
* `f'{name}: {params[idx]:.4}'`
    * `f`：以前説明したf-string
    * `{name}`：`exog`の変数名を代入する。
    * `{params[idx]:.4}`
        * `params[idx]`：`idx`番目の`params`を代入する。
        * `:.4`：小数点第4位までを表示する。

---
「手計算」の結果は`linearmodels`を使う結果と同じになることを下で確認する。

### `linearmodels`を使う計算

#### EntityEffects

`linearmodels`には`PanelOLS`モジュールがあり，その関数`from_formula()`を使うことにより，`statsmodels`同様，回帰式を`y ~ x`の文字列で書くことが可能となる。その際，次の点に注意すること。
* 固定効果推定をする場合，回帰式に`+ EntityEffect`を含める。
    * このオプションにより変数の平均からの乖離は自動で計算されることになる。
    * `+ EntityEffect`を含めないと`PooledOLS`（通常のOLS）と等しくなる。


まず`wagepan`を`MultiIndex`化する。これにより`linearmodels`を使いFD推定可能となる。

wagepan = wagepan.set_index(['nr','year'],drop=False)

wagepan.head()

wagepan.info()

次に`PanelData`オブジェクトに変換しデータの特徴を調べる。

wagepanp = PanelData(wagepan)
wagepanp.shape

* 44: 変数の数
* 8: 期間数（年）
* 545：観察単位の数（人数）

次に，balanced もしくは unbalanced data set かを確認する。

(wagepanp.count()==wagepanp.nobs).all().all()

このデータ・セットはbalancedだが，unbalanced だったとしても，固定効果モデルの考え方や以下で説明するコードは変わらない。

---
実際に回帰式を書くことにする。使い方は`statsmodels`と似ている。
* `PanelOLS`モジュールの関数`.from_formula`を使い次のように引数を指定する。

$$\text{.from_formula}(\text{回帰式}, \text{データ})$$

* `EntityEffects`を加える。
* 定数項を入れたい場合は，`1`を回帰式に追加する。入れなければ定数項なしの推定となる。

* 以下では時間ダミー`C(year)`が入るので入れない。

formula_fe = 'lwage ~ married + union + expersq \
                      +d81+d82+d83+d84+d85+d86+d87 + EntityEffects'

* 固定効果モデルのインスタンスの作成

mod_fe = PanelOLS.from_formula(formula_fe, data=wagepan)

* `statsmodels`と同じように，そこから得た結果にメソッド`.fit()`を使い計算し結果が返される。

result_fe = mod_fe.fit()

＜結果の表示方法＞
1. `res_fe`を実行。
1. `res_fe`に関数`print()`を使うと見やすい。
1. `res_fe`には属性`summary`が用意されているが，表示方法1と同じ。
1. `summary`には属性`tables`があり，２つの表がリストとして格納されている。
    * `tables[0]`：検定統計量の表（`print()`を使うと見やすくなる）
    * `tables[1]`：係数の推定値やp値などの表（`print()`を使うと見やすくなる）

print(result_fe.summary.tables[1])

（結果）
* `exper**2`の係数が負で統計的有意性が高いのは，賃金に対して経験の効果は低減することを示している。
* `married`の係数は正であり，優位性が全くないわけではない。賃金の既婚プレミアムと呼ばれるものである。
* `union`は労働組合の影響を示しているが，予測通りである。

$R^2$を表示してみる。

print(result_fe.summary.tables[0])

この表にある$R^2$について説明する。
* $R^2$＝$R^2(\text{Within})$：(式３)を推定した際の$R^2$である（$\ddot{y}_i$が$\ddot{x}_i$によってどれだけ説明されたかを示す）。
* $R^2(\text{Between})$：(式２)を推定した際の$R^2$である（$\hat{y}_i$が$\hat{x}_i$によってどれだけ説明されたかを示す）。
* $R^2(\text{Overall})$：(式１)を推定した際の$R^2$である（${y}_i$が${x}_i$によってどれだけ説明されたかを示す）。

#### TimeEffects

上の推定式では時間ダミー変数として使い，観察単位全てに共通な時間的な影響を捉えた。具体的には，インフレにより賃金は変化するが，その変化は全ての労働者には対して同じであり，その効果を時間ダミー変数が捉えている。それを**時間効果**と呼ぶ。このような時間ダミー変数を加えた理由は，時間効果が他の変数（例えば，`married`）の係数を「汚さない」ようにするためであり，よりピュアの効果を推定するためである。一方，`linearmodels`では，わざわざ時間ダミー変数を作らずとも`TimeEffects`を回帰式に追加することにより，時間効果を自動的に「排除」することができる。

formula_fe2 = 'lwage ~ married + union + expersq + TimeEffects + EntityEffects'
result_fe2 = PanelOLS.from_formula(formula_fe2, data=wagepan).fit()
print(result_fe2)

`result_fe`と同じ結果を確認できる。$R^2$の値は少し変わっているが，これは時間ダミーを入れて計算している訳ではないためである。

## ダミー変数モデルとしての固定効果モデル

固定効果推定量は，他の推定方法でも計算することができる。その１つがダミー変数推定である。アイデアは簡単で，観察単位のダミー変数を使い異質性を捉えるのである。推定方法も簡単で，（式１）に観察単位のダミー変数を加えて通常のOLS推定をおこなうだけである。

上で使った`wagepan`を使い推定する。まず，観察単位のダミー変数として推定式に`C(nr)`を追加する

formula_dum = 'lwage ~  1 + married + union + expersq \
                        +d81+d82+d83+d84+d85+d86+d87 + C(nr)'

`PooledOLS`モジュールの関数`from_formula`を使って推定式を定義する。ここで`PooledOLS`とは，`statsmodels`で使う通常のOLS推定と同じである。

result_dum = PooledOLS.from_formula(formula_dum, data=wagepan).fit()

`nr`のダミー変数が544あるため，そのまま結果を表示せずに，以下のセルにあるフィルターを使いメインの変数だけを表示する。下のコードの`filter`で使う（`result_dum`に続く）属性・メソッドについて：
* `params`：パラメーターを取得する属性
* `index`：`params`のインデックスを取得する属性
* `str`：`index`にある文字列を操作可能なオブジェクトとして取得する属性
* `contains()`：引数の文字列が文字列の中にある場合`True`を返すメソッド
* `( )`：括弧の中を先に評価するという意味
* `tolist()`：リストに変換するメソッド

filter = (result_dum.params.index.str.contains('nr') == False).tolist()
result_dum.params[filter]

３つの変数`I(exper**2)`，`married`，`union`の係数は固定効果モデルと等しいことが確認できる。

t値とp値の表示には次のコードを使う。
* t値：`result_dum.tstats[filter]`
* p値：`result_dum.pvalues[filter]`

この場合，`educ`等の時間に対して不変の変数も推定式に加えることが可能である。一方，この方法はパラメーターの数は544+11=555あり，自由度が非常に低くなるのが欠点である。


（注意）

この推定式に時間に対して不変の変数（例えば，`educ`，`blac`，`hisp`，`exper`）を追加すると推定できない（エラーが発生する）。理由は，それらの変数は`nr`ダミー変数と完全に同じ動きをするためである。

## ランダム効果モデル

### 説明

ここでは，パネル・データを使い推定するランダム効果モデル（Random Effects Model; RE Model）を解説する。もう一度（式１）を考えよう。

（仮定） $a_i$は観察単位 $i$ に対して一定であるが，$i$によってその値は異なる。

＜固定効果モデルの仮定＞
* $\text{Cov}\left(a_ix_{it}\right)\neq 0,\quad t=1,2,...,T$

＜ランダム効果モデルの仮定＞
* $\text{Cov}\left(a_ix_{it}\right)=0,\quad t=1,2,...,T$

---
この違いを念頭に（式１）を次式に書き換える。

$$y_{it}= \beta_0 + \beta_1x_{it}+e_{it}\qquad\qquad\left(\text{式４}\right)$$

ここで，

$$e_{it}=a_i+u_{it}$$

しかし，

$$\text{Corr}\left(e_{it},e_{is}\right)\neq 0,\quad t\neq s$$

となることが示せる。即ち，誤差項が自己相関することになる。

＜含意＞
* $a_i$と$x_{it}$に相関がないため次の推定方法で一致性を満たす推定量を計算できる。
    * １つの時点でお横断面データを使いOLS推定（でも他のデータはどうする？）
    * 全てのデータを使い何の区別もなくプールするPooled OLS推定（しかしパネル・データの特性を有効利用していない）。
* 時系列データを扱う場合，自己相関によりOLS推定量は効率性が低くなる。これは横断面データを扱う際の不均一分散が引き起こす問題と似ている。
* １階差分推定や固定効果推定を使う必要はなく，使うと効率性が低い推定量となる。
* 従って，問題は次の点：
    * より効率性が高い推定方法はどのようなものか。

---
その方法がランダム効果推定と言われるもので，固定効果のように，変数を平均からの**部分的**な乖離に変換することにより可能となる。具体的には，次式が推定式となる。

$$\overset{\circ}{y}_{it}=\beta_0(1-\theta)+\beta_1\overset{\circ}{y}_{it}+\overset{\circ}{e}_{it}\qquad\qquad\left(\text{式５}\right)$$

ここで
* $\overset{\circ}{y}_{it}=y_{it}-\theta\bar{y}_i$
* $\overset{\circ}{x}_{it}=x_{it}-\theta\bar{x}_i$
* $\overset{\circ}{e}_{it}=e_{it}-\theta\bar{e}_i$

は変数の平均からの部分的な乖離であり，乖離の度合いを決める変数$\theta$は

 $$\theta = 1-\sqrt{\frac{\sigma_u^2}{\sigma_u^2+T\sigma_a^2}}$$

と定義される。$\sigma_u^2$は$u_{it}$の分散，$\sigma_a^2$は$a_{i}$の分散である。



（直感）
* $\sigma_a^2=0$の場合$\theta=0$となり，$a_i$は$i$に対して一定であり観察単位の異質性はないということである。その場合は，通常のOLSで推定することがベストとなる。
* $\sigma_a^2$が増加すると$\theta$は大きくなり，より大きな平均からの乖離が必要となる。極端な場合，$\sigma_a^2$が無限大に近づくと，$\theta=1$となり$\overset{\circ}{y}=\ddot{y}$となる。即ち，固定効果モデルはランダム効果モデルの極端な場合と解釈できる。

（注意）
* $\theta$は事前にはわからないため推定する必要がある（`linearmodels`が自動的に計算する）。

（良い点）
* 時間に対して不変の説明変数があってもその係数を推定できる。
* $\text{Cov}\left(a_ix_{it}\right)= 0$が正しければ，推定量は一致性を満たす。しかし，不偏性は満たさない。

### 推定

`exper`，`educ`，`black`，`hisp`を加えて回帰式を定義する。

（注意）

時間ダミー変数の代わりに`TimeEffects`を使わないように。入れることができますが，そのような仕様になっていません。

formula_re = 'lwage ~ 1 + married + union + expersq \
                        + exper + educ + black + hisp \
                        +d81+d82+d83+d84+d85+d86+d87'

`RandomEffects`のモジュールにある関数`from_formula`を使い計算する。

result_re = RandomEffects.from_formula(formula_re, data=wagepan).fit()

結果の表示。

print(result_re.summary.tables[1])

結果の解釈は下でする。

RE推定では$\theta$が重要な役目を果たすが，その値は`result_re`の属性`theta`を使うことにより`DataFrame`の形で表示できる。
* balanced panel dataの場合，`theta`の値は一意で決まる。
* unbalanced panel dataの場合，`theta`は観察単位毎に計算される。

result_re.theta.iloc[0,:]

上で説明したが$\theta$は$u_a$と$u_it$の分散である$\sigma_a^2$と$\sigma_u^2$に依存する。それらの値は，属性`variance_decomposition`を表示できる。以下の返り値の内容：
* `Effects`：$\sigma_a^2$
* `Residual`：$\sigma_u^2$
* `Percept due to Effects`：$\dfrac{\sigma_a^2}{\sigma_a^2+\sigma_u^2}$

result_re.variance_decomposition

## 相関ランダム効果モデル

### 説明

相関ランダム効果モデル（CREモデル）は，固定効果モデルとランダム効果モデルの中間的な位置にあり，両方を包含している。（式１）を考えよう。更に，観察不可能な固定効果$a_i$は説明変数と次の関係にあると仮定する。

$$a_i = \alpha + \gamma\bar{x}_{it} + r_i\qquad\text{(式５)}$$

* $\bar{x}_i=\dfrac{1}{T}\sum_{t=1}^Tx_{it}$は説明変数の平均
* $\gamma$は$a_i$と$x_{it}$の相関関係を捉える係数
* $r_i$は説明変数$x_{it}$と相関しないと仮定，即ち，$\text{Cov}\left(r_i\bar{x}_{it}\right)$

（式５）を（式１）に代入すると次式を得る。

$$y_{it}=\alpha+\beta x_{it} + \gamma\bar{x}_i + v_{it}\qquad\text{(式６)}$$

ここで

$$v_{it}=r_i + u_{it}$$

（含意）
* $\text{Cov}\left(r_i,\bar{x}_{it}\right)\;\Rightarrow\;\text{Cov}\left(v_i,\bar{x}_{it}\right)$
* REモデルと同じ構造となっており，違いは$\bar{x}_i$が追加されている。
    * `linearmodels`の`RandomEffects`モジュールが使える。
* 次の結果が成立する。

    $$\hat{\beta}_{FE}=\hat{\beta}_{CRE}$$
    
    * $\hat{\beta}_{FE}$：固定効果推定量
    * $\hat{\beta}_{CRE}$：相関ランダム効果推定量
    * この結果は，時間に対して不変な変数（例えば，`black`）を**含めても**成立する

### 推定

まず $\bar{x}_i$ を計算し，それを`wagepan`に追加する。そのために次の関数を定義する。

（解説）

* (1)：関数の引数
    * `dframe`：データフレーム
    * `ori_col`：平均を計算したい列
    * `new_col`：計算した平均を入れる列
* (2)：`ori_col`をグループ化し，グループ名とグループ平均からなる辞書の作成
    * `groupby(level=0)`：行の第１インデックスに従ってグループ化
    * `mean()`：グループ平均の計算
    * `to_dict()`：行の第１インデックスにあるグループ名を`key`，グループ平均を`value`にする辞書の作成
* (3)：行の第１インデックスに該当するグループ平均が並ぶリストの作成
    * `index.get_level_values(0)`：行の第１インデックスの値を取得
    * `to_series()`：`series`に変換
    * `map(dict)`：`dict`の内容に従って上の`series`の値をグループ平均に入れ替える
    * `tolist()`：リストに変換
* (4)：`dframe`にグループ平均が並ぶ新しい列が追加し，そのラベルを`new_col`とする
* (5)：`DataFrame`を返す

def add_col_mean(dframe, ori_col, new_col):  # (1)
    
    dict = dframe.groupby(level=0)[ori_col].mean().to_dict()  # (2)
    mean = dframe.index.get_level_values(0).to_series().map(dict).tolist()  # (3)
    dframe.loc[:,new_col] = mean  # (4)
    
    return dframe   # (5)

この関数を使い，`married`，`union`，`expersq`の平均を計算し`wagepan`に追加する。

（コメント）`exper`は含めない。

wagepan = add_col_mean(wagepan, 'married', 'married_mean')
wagepan = add_col_mean(wagepan, 'union', 'union_mean')
wagepan = add_col_mean(wagepan, 'expersq', 'expersq_mean')

CRE推定と結果の表示

formula_cre = 'lwage ~ 1 + married + union + expersq \
                         + married_mean + union_mean + expersq_mean \
                         +d81+d82+d83+d84+d85+d86+d87'

result_cre = RandomEffects.from_formula(formula_cre, data=wagepan).fit()

print(result_cre)

### ２つの利点

CREモデルの２つの利点を解説する。

#### FE対RE検定

FEモデルとREモデルのどちらが適しているかを調べることができるHausman検定というものがある。CREモデルを使うことにより，同様の検定が簡単に行える。（式６）を考えよう。
* $\gamma=0$の場合，REモデルの（式５）と同じになり，REモデルが妥当ということになる。
* $\gamma\neq 0$の場合，$a_i$と$\bar{x}_{i}$は相関することになり，これは$\text{Cov}\left(a_i,{x}_{it}\right)\neq 0$を意味し，REモデルが妥当な推定方法となる。

この考えを利用して，次の帰無仮説と対立仮説のもとで$\gamma$の優位性を調べる。
* $\text{H}_0:\;\text{Cov}\left(a_i,x_{it}\right)=0$
* $\text{H}_a:\;\text{Cov}\left(a_i,x_{it}\right)\neq 0$

（コメント）
* $\gamma=0$を棄却できれば上の$\text{H}_0$を棄却できる。
* 平均の変数が$k$ある場合は，$\gamma_1=\gamma_2=\cdots=\gamma_k=0$を検定する。

`result_cre`のメソッド`wald_test()`を使う。引数には文字列で指定する。

# 検定する係数の値を設定する
restriction = 'married_mean = union_mean = expersq_mean = 0'

# 検定結果を表示する
result_cre.wald_test(formula=restriction)

p値が非常に小さいので，帰無仮説は棄却できる。従って，FEモデルが妥当だと結論づけることができる。

#### 一定な変数を含める

（式６）に時間に対して一定は変数（$z_{i}$）を含めて

$$y_{it}=\alpha+\beta x_{it}+\gamma\bar{x}_i+\delta z_{i} +v_{it}\qquad\text{(式６)}$$

をRE推定しても次の結果は成立する。

$$\hat{\beta}_{FE}=\hat{\beta}_{CRE}$$

---
この結果を利用して，以下では次の変数を加えて回帰式を設定する。
* 時間に対して一定な変数
    * `educ`, `black`，`hisp`
* 一定ではないが，FEにいれると推定できなかった変数
    * `exper`

formula_cre2 = 'lwage ~ 1 + married + union + expersq \
                          + exper + educ + black + hisp \
                          + married_mean + union_mean + expersq_mean \
                          +d81+d82+d83+d84+d85+d86+d87'

result_cre2 = RandomEffects.from_formula(formula_cre2, data=wagepan).fit()

print(result_cre2.summary.tables[1])

もう一度，FE対RE検定を行ってみよう。検定統計量（Statistics）は減少したが，以前帰無仮説は高い優位性で棄却できる。

result_cre2.wald_test(formula=restriction)

## モデルの比較

パネル・データを扱う場合の通常のアプローチは，使える推定法を使いその結果を比べることから始める。以下では以下のモデルを比べる。
* 通常のOLS
* 固定効果モデル
* ランダム効果モデル
* 相関ランダムモデル

### OLS

`linearmodels`のモジュール`PooledOLS`では，観察単位や時間の区別なく全てのデータをプールしてOLS推定する。これは通常のOLSと同じ推定法と等しい。`PooledOLS`の関数`from_formula`を使い，以下のように推定する。

formula_pool = 'lwage ~ 1 + married + union + expersq \
                        + exper + educ + black + hisp \
                        +d81+d82+d83+d84+d85+d86+d87'

result_pool = PooledOLS.from_formula(formula_pool, data=wagepan).fit()

print(result_pool.summary.tables[1])

### 比較表の作成

今までの推定結果を表にまとめるために，`linearmodels`の関数`compare`を使う。

表作成の順番：
1. `key`が表示したい推定方法の名前，`value`がその上で得た推定結果となる辞書を作る
1. その辞書を`compare`の引数としてつかう。

res = {'Pooled OLS':result_pool,
       'FE': result_fe,
       'RE': result_re,
       'CRE': result_cre2
      }

# compare(res)

このままでも良いか，この方法では推定結果を表示する順番を指定できない。例えば，`OLS`，`FE`，`RE`，`CRE`の順番で左から並べたいとしよう。その場合，`collections`パッケージにある関数`OrderedDict`をつかう。`{}`の中で並べた順番をそのまま維持してくれる関数である。

res_ordered = OrderedDict(res)
print(compare(res_ordered))

* `married`
    * OLSでの結婚プレミアムは10％以上あるが，FEでは半減している。これは観察単位の異質性$a_i$に生産性が含まれており，「生産性が高い人（高い$a_i$）は，結婚している可能性が高い」という考えと一貫性がある。即ち，$\text{Cov}\left(a_i,x_{it}\right)>0$となり，これによりOLSでは以下のようなバイアスが発生すると解釈できる。
        * 既婚 $\Rightarrow$ 生産性（$a_i$）が高い $\Rightarrow$ 賃金が上振れしやすい
        * 未婚 $\Rightarrow$ 生産性（$a_i$）が低い $\Rightarrow$ 賃金が下振れしやすい
    * FE推定法が示しているのは，そのようなバイアスを取り除いても，結婚プレミアムは存在する。考えられる理由は：
        * 結婚は生産性を上昇させる
        * 結婚は安定した生活を意味し，それに対して企業はより高い賃金を払う
    * REは推定値はFEに比較的に近い。これは$\hat{\theta}>0.5$の値にも現れている。
* `union`
    * FEの値は労働組合の賃金に対する影響力を示している。
    * OLSとFEの推定値を比較すると，約0.1減少している。OLSは労働組合の影響力を過大評価しており，観察単位の異質性が大きく働いていることがわかる。
* `educ`，`black`，`hisp`
    * OLSもREも推定値は似ている。

### FD vs. FE

1階差分モデルと固定効果モデルを比較する。
* $T=2$の場合，FDとFEは同じ（if there is intercept in FE）
* $T\geq 3$の場合：
    * GM仮定に対応する仮定の下ではFDもFEも不偏性・一致性を満たす。
    * 誤差項に系列相関がない場合，FEの方が効率性が高い
    * 誤差項の系列相関がある場合，FDの方が良い。
        * 例えば，誤差項がランダム・ウォークの場合$\Delta u_{it}$は系列相関はない。
        * FD推定をして$\Delta u_{it}$を検定する。
    * $N$が少なく$T$が大きい場合（例：$N=20$と$T=30$），時系列の特性が強くなるので，FDの方が良い
* 実証研究では，FDとFEの結果の両方を報告すること。

### FE vs. RE

固定効果モデルとランダム効果モデルを比較する。
1. $a_i$はランダムか？
    * 経済学の場合，説明変数は何らかの選択の結果の場合が多い。さらに，その選択が観察単位の特徴に依存する場合，$\text{Cov}\left(a_ix_{it}\right)\neq 0$となり，FEモデルの方が適切となる。
    * 都道府県データのような場合，「大きな」母集団からランダムに抽出された説明変数とはならないかも知れない。むしろ，都道府県の切片がランダムではなく，単に異なると仮定する方が自然かも知れない。
1. Hausman検定であれ他の検定であれ，間違う確率は存在する。以下では間違った場合どうなるかを考えた。
    * $\text{Cov}\left(a_ix_{it}\right)= 0$，即ち，REモデルが正しい場合：
        * 誤差項の自己相関があるが，FE推定量は不偏性を満たす
    * $\text{Cov}\left(a_ix_{it}\right)\neq 0$，即ち，FEモデルが正しい場合：
        * GM仮定４が満たされないため，RE推定量は不偏性を満たさない

Pros for RE
1. FEの場合，時間に対して変化しない変数の係数を推定できない
2. 回帰式が非線形の場合（例えば，Probit），FEでは対応できない。


（結論）一般的にFEの方が適切な場合が多いのではないか。

## シミュレーション

（目的）真のモデルでは観察単位の固定効果がある場合を考え，FE推定量とRE推定量を比較する。

単回帰を想定する。以下を真の値として設定する。

b0 = 1  # 定数項
b1 = 2  # スロープ係数

### 推定値の計算

シミュレーションの基本パラメータ等の設定

N = 100  # 観察単位の数
T = 5  # 時間数
ai = np.linspace(0,10,N)  # 観察単位の異質性

`for`ループによる`DataFrame`の作成。

df_sim = pd.DataFrame()  # 空のDataFrame

for (idx,a) in enumerate(ai):
    
    x = norm.rvs(a,1,size=T)  # T個のaが平均となるランダムな数
    u = norm.rvs(0,1,size=T)  # 誤差項
    y = b0 + b1*x + a + u  # 被説明変数
    df_idx = pd.DataFrame({'id':[idx]*T,    # 観察個体のID
                         'time':np.array(range(T))+2000,
                         'Y':y,
                         'X':x,
                         'ai':[a]*T})
    df_sim = pd.concat([df_sim,df_idx])

`DataFrame`の微調整。

# id と time の列を 整数型に変換（省いても問題ない）
df_sim['id'] = df_sim['id'].astype(int)
df_sim['time'] = df_sim['time'].astype(int)

# MultiIndex化
df_sim = df_sim.set_index(['id','time'])

df_sim.head()

固定効果モデルによる推定。

form_sim_fe = 'Y ~ X + EntityEffects'

sim_fe = PanelOLS.from_formula(form_sim_fe, data=df_sim).fit()

print(sim_fe.summary.tables[1])

ランダム効果モデルによる推定。

form_sim_re = 'Y ~ 1 + X'

sim_re = RandomEffects.from_formula(form_sim_re, data=df_sim).fit()

print(sim_re.summary.tables[1])
print('theta:', sim_re.theta.iloc[0,:].values)

$\text{Cov}\left(a_i,x_{it}\right)>0$により上方バイアスが発生している。

相関ランダム効果モデルによる推定。

まず，観察単位の`X`の平均の列を追加する。

df_sim = add_col_mean(df_sim, 'X', 'X_mean')

form_sim_cre = 'Y ~ 1 + X + X_mean'

sim_cre = RandomEffects.from_formula(form_sim_cre, data=df_sim).fit()

print(sim_cre.summary.tables[1])

通常のOLS（Pooled OLS）による推定はバイアスが発生する。

form_sim_pool = 'Y ~ 1 + X'

sim_pool = PooledOLS.from_formula(form_sim_pool, data=df_sim).fit()

print(sim_pool.summary.tables[1])

### 推定値の分布

基本的に上のシミュレーションのコードを応用する。

N = 100  # 観察単位数
T = 5  # 年数
R = 100  # シミュレーションの回数
ai = np.linspace(0,10,N)  # 観察単位の異質性

（下のコードについて）
* ランダム効果モデルの推定には`linearmodels`を使っている。固定効果モデルと通常のOLS推定にも`linearmodels`を使うと必要がない統計量も計算するため計算に時間が掛かる。少しでも計算時間を縮めるために「手計算」をする。

bhat_fe_list = []  # FE推定値を入れるための空のリスト
bhat_re_list = []  # RE推定値を入れるための空のリスト
bhat_pool_list = []  # Pooled OLS推定値を入れるための空のリスト

for _ in range(R):  # Rの値は下のコードで使わないので"_"に設定する
    
    df = pd.DataFrame()  # 空のDataFrame

    # データの生成
    for (idx,a) in enumerate(ai):
        x = norm.rvs(a,1,size=T)  # T個のaが平均となるランダムな数
        u = norm.rvs(0,1,size=T)  # T個の誤差項
        y = b0 + b1*x + a + u  # T個の被説明変数
        df_idx = pd.DataFrame({'id':[idx]*T,      # DataFrameへ表の追加
                             'time':np.array(range(T))+2000,
                             'Y':y,
                             'X':x,
                             'ai':[a]*T})
        df = pd.concat([df,df_idx])
        
    # RE推定
    df_re = df.set_index(['id','time'])  # MultiIndex化
    form_sim_re = 'Y ~ 1 + X'
    sim_re = RandomEffects.from_formula(form_sim_re, data=df_re).fit()
    bhat_re_list.append(sim_re.params[1])

    # FE推定
    df_fe = df.loc[:,['Y','X']] - df.groupby('id')[['Y','X']].transform('mean')
    Yfe = df_fe.loc[:,'Y'].values
    Xfe = df_fe.loc[:,'X'].values[:,None]  # [:,None]は(N*T,1)の行列に変換
    bhat_fe = (np.linalg.inv(Xfe.T@Xfe)@Xfe.T@Yfe)[0]
    bhat_fe_list.append(bhat_fe)
    
    # Pooled OLS推定
    c = np.ones(N*T)
    Xpool = np.stack([c, df.loc[:,'X'].values], axis=1)
    Ypool = df.loc[:,'Y'].values
    bhat_pool = (np.linalg.inv(Xpool.T@Xpool)@Xpool.T@Ypool)[1]
    bhat_pool_list.append(bhat_pool)

分布の図示

xx=np.linspace(1.6,3.1,num=100)  # 図を作成するために横軸の値を設定

kde_model_fe=gaussian_kde(bhat_fe_list)  # FE推定量のカーネル密度関数を計算

kde_model_re=gaussian_kde(bhat_re_list)  # RE推定量のカーネル密度関数を計算

kde_model_pool=gaussian_kde(bhat_pool_list)  # Pooled OLS推定量のカーネル密度関数を計算

plt.plot(xx, kde_model_fe(xx), 'g-', label='FE')  # FE推定量の分布プロット
plt.plot(xx, kde_model_re(xx),'r-', label='RE')  # RE推定量の分布プロット
plt.plot(xx, kde_model_pool(xx),'k-', label='Pooled OLS')  # RE推定量の分布プロット
plt.axvline(x=b1,linestyle='dashed')  # 真の値での垂直線
plt.ylabel('Kernel Density')  # 縦軸のラベル
plt.legend()  # 凡例
pass

## 標準誤差の問題

(式３)の固定効果モデルを考えよう。パネル・データの場合，次の2つの問題が発生する場合がある。
* 残差の不均一分散
* 残差の自己相関

これらの問題が発生しても，基本的な仮定のもとでFE推定量は不偏性と一致性を満たす。しかし，係数の標準誤差は有効ではなくなり検定が無効となる。その対処方法として**クラスター頑健的推定**を使う。これは不均一分散の章で説明した不均一分散頑健的推定を自己相関に拡張し，パネル・データ用に考案された推定と理解すれば十分である。以下では，その使い方を説明する。

### 分散の確認

`linearmodels`には残差の均一分散を調べるBreusch-Pagan検定やWhite検定をおこなうメソッドは用意されていない。ここでは`statsmodels`を使いWooldridge (2016,p.253)で説明されているWhite検定に基づく検定をおこなう。次に必要な変数を作成する。

# 残差
u_hat = result_fe.resids.values.flatten()

# 被説明変数の予測値
y_hat = result_fe.fitted_values.values.flatten()

# DataFrameの作成
df_white = pd.DataFrame({'u_hat':u_hat,'y_hat':y_hat})

（上のコードの説明）
* `resids`は残差を`DataFrame`として取得する属性
* `fitted_values`は予測値を`DataFrame`として取得する属性
* `values`は`DataFrame`を`array`として返す属性
* `flatten()`は`array`が2次元になっているのを1次元に変換するメソッド
    * `array([[..],[..],...[...]])`を`array([....])`に変換する。

検定に使う式

$$\hat{u}^2=\beta_0+\beta_1\hat{y}+\beta_2\hat{y}^2+e$$

* $\text{H}_0：\beta_1=\beta_2=0$（均一分散）
* $\text{H}_A：$帰無仮説は成立しない

form_white = 'I(u_hat**2) ~ y_hat + I(y_hat**2)'

res_white = ols(form_white, data=df_white).fit()

print(res_white.summary().tables[1])
print('F検定のp値：',res_white.f_pvalue)

帰無仮説は棄却できない。

次に図を使い確認する。

b0 = res_white.params[0] # beta0
b1 = res_white.params[1] # beta1
b2 = res_white.params[2] # beta2

xx = np.linspace(min(y_hat), max(y_hat), 100) # x軸の値
z = b0 + b1*xx + b1*xx**2  # 検定に使った式

plt.scatter(y_hat, u_hat**2)  # u_hat, y_hatの散布図
plt.plot(xx, z, 'red', linewidth=3)  # 検定の式の曲線
plt.xlabel('y_hat')
plt.ylabel('u_hat^2')
pass

（解説）
* 上の検定で推定した式は赤い線である。殆ど平行になっているため帰無仮説を棄却できなかった。
* 図の中で`y_hat`の値が`-0.2`を境に`u_hat`の変動はより大きくなっており，不均一分散の疑いが高い。

### 対処方法

＜使い方＞
* メソッド`fit()`に以下の引数を指定する。
    * 不均一分散だけの場合：`cov_type='clustered', cluster_entity=True`
    * 不均一分散と系列相関の場合：`cov_type='clustered', cluster_entity=True, cluster_time=True`


（注意）
* 係数の推定値は変わらない。
* 係数の標準誤差だけが修正され，有効な検定ができるようになる。


`wagepan`を使って上で推定した式にクラスター頑健的推定を使う。

＜不均一分散かだけの場合＞

mod_fe_clus1 = PanelOLS.from_formula(formula_fe, data=wagepan)

res_fe_clus1 = mod_fe_clus1.fit(cov_type='clustered', cluster_entity=True)

print(res_fe_clus1.summary.tables[1])

＜不均一分散と系列相関の場合＞

mod_fe_clus2 = PanelOLS.from_formula(formula_fe, data=wagepan)

res_fe_clus2 = mod_fe_clus2.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

print(res_fe_clus2.summary.tables[1])