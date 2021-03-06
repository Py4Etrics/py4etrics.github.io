# `linearmodels`

If you come here without expecting Japanese, please click [Google translated version](https://translate.google.com/translate?hl=&sl=ja&tl=en&u=https%3A%2F%2Fpy4etrics.github.io%2F16_linearmodels.html) in English or the language of your choice.

---

import pandas as pd
from linearmodels.panel.data import PanelData
from linearmodels.panel import FirstDifferenceOLS
import wooldridge
from see import see

# 警告メッセージを非表示
import warnings
warnings.filterwarnings("ignore")

## 説明

`linearmodels`は`statsmodels`を補完する目的として開発されている。主に，パネルデータ，操作変数法を使った推定法やGMMを扱う場合には非常に重宝するパッケージである。しかし，`linearmodels`は`statsmodels`の両方を使う上で以下の点に注意する必要がある。
* 推定結果などのメソッドや属性が共通化されているわけではない。次の表に３つの例を挙げる。


|              | 推定結果の<br> 表を表示 | 残差を<br> 表示する<br> メソッド | 標準誤差を<br> 取得する<br> 属性 |
|-------------:|:-----------------------:|:--------------------------------:|:--------------------------------:|
| statsmodels  | .summary()              | .resid                           | .bse                             |
| linearmodels | .summary                | .resids                          | .std_errors                      |


* `statsmodels`も`linearmodels`も回帰式を文字列で指定できるが，定数項を指定する方法が異なる。
    * `statsmodels`では，定数項は自動的に追加される，定数項を省く場合は`-1`を追加する。
    * `linearmodels`では，定数項は自動的に追加されない。定数項を入れる場合は`1`を追加する。
* `fit()`メソッドの挙動も共通化されていない。
    * `linearmodels`の`fit()`に何のオプションも指定せずにOLS推定すると係数の推定量は同じだが，標準誤差や$t$値などが異なる。同じにするためには次のように２つのオプションを設定しなくてはならない。
    ```
    .fit(cov_type='unadjusted', debiased=True)
    ```
    * `cov_type`は不均一分散頑健共分散行列推定のオプション
        * デフォルトは`robust`（不均一分散頑健的共分散行列推定）で`statsmodels`の`HC1`と等しい。
    * `debiased`は共分散行列推定の自由度のオプション（小標本の場合の調整）
        * デフォルトは`False`

**（注意）**

以下では`.fit()`のオプションは指定せず，デフォルトのまま議論を続ける。

---
以下では`linearmodels`を使うが，そのためには`DataFrame`を`MultiIndex`に変換する必要がある。以下では，まず`MultiIndex`について説明し，その後に`linearmodel`にある`PanelData`関数について説明する

## `Pandas`の`MultiIndex`

### 説明

パネル・データを扱うために必要な`Pandas`の`MultiIndex`について説明する。`MultiIndex`とは行や列のラベルが階層的になった`DataFrame`や`Series`を指す。以下では，`DataFrame`の行における`MultiIndex`を説明する。

まずデータを読み込む。

# url の設定
url = 'https://raw.githubusercontent.com/Haruyama-KobeU/Haruyama-KobeU.github.io/master/data/data4.csv'

# 読み込み
df = pd.read_csv(url)
df

行・列ともにラベルの階層は１つずつとなっている。`set_index()`を使い行に`MultiIndex`を作成するが，引数に

$$\left[\text{第０id},\text{第１id}\right]$$

とし階層インデックス化する。ここでパネル・データ分析をする上で以下のルールに従うことにする。
* 第０id：観察単位（例えば，消費者，企業，国）
* 第１id：時間（例えば，年，四半期）

次の例では`country`と`year`の行をそれぞれ第０インデックス，第１インデックスに指定する。

df = df.set_index(['country', 'year'])#.sort_index()
df

階層インデックスが綺麗に並んでいるが，元のデータの並び方によっては階層インデックスが期待通りに並ばない場合がありえる。その場合は，メソッド`sort_index()`を使うと良いだろう。

---
`MultiIndex`を解除するにはメソッド`.reset_index()`を使う。

df.reset_index()

### 要素，行，列の抽出

`MultiIndex`のまま要素・列・行の抽出およびスライシングには様々な方法があり，複雑である。特に，スライシングをしたい場合，一番簡単なのは`reset_index()`で通常の`DataFrame`に戻し，スライシングし新たな`DataFrame`を作成するだけでも十分であろう。

以下では，`.loc[]`を使い`MultiIndex`のままでの抽出方法について簡単に説明する。その際，以下のルールは変わらない。

$$.\text{loc}\left[\text{行の指定},\text{列の指定}\right]$$

ただ，`行の指定`にリストやタプルを使うことになる（`列の指定`も同じ）。

他の方法については[このサイト](https://note.nkmk.me/python-pandas-multiindex-indexing/)と[このサイト](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)が参考になる。

#### １つの観察単位の抽出

１つの要素を抽出する場合はタプルを使う。例えば，日本の2001年の`gdp`を抽出したい場合。

df.loc[('Japan',2001), 'gdp']

#### 行の抽出

上の例で列の指定を`:`にすると，指定した行に対して全ての列を抽出できる。

df.loc[('Japan',2001), :]

この場合，特定の列に対してスライシングも可能。

df.loc[('Japan',2001), 'gdp':'con']

指定した行に対して個別に複数列を抽出したい場合は，タプルを使う。

df.loc[('Japan',2001), ('gdp','con')]

複数行の抽出にはリストで指定する。

df.loc[(['Japan','UK'],[2001,2002]), :]

#### 第０インデックスの観察単位の全て

第０インデックスにある，ある観察単位の全てのデータだけを抽出したい場合は，通常の`Pandas`の場合と同じ。

df.loc['Japan', :]

複数の場合。

df.loc[['Japan','UK'], :]

#### 列の抽出

通常の`Pandas`と同じ。`Series`を返す場合。

df.loc[:,'gdp']

`[]`を使うと，`DataFrame`として抽出できる。

df.loc[:,['gdp']]

複数列抽出の場合。

df.loc[:,['gdp','inv']]

スライシングも使える。

df.loc[:,'gdp':'con']

#### 第１インデックスのある年だけの抽出

一番簡単な方法は`reset_index()`を使い今まで習った関数を使う。

df.reset_index().query('year == 2000')

複数年の場合。

df.reset_index().query('year in [2000,2002]')

上と同じ結果。

df.reset_index().query('year not in [2001]')

## `linearmodels`の`PanelData`


`linearmodels`では`MultiIndex`化された`DataFrame`をそのまま読み込み推定することができる。一方で，`linearmodels`の関数`PanelData`を使い`MultiIndex`化された`DataFrame`を`PanelData`オブジェクトに変換すると分析に必要な計算を簡単にできるようになる。必須ではないが，知っておいて損はしない関数である。

まず`df`を`PanelData`オブジェクトに変換する。

dfp = PanelData(df)
dfp

### 属性とメソッド

まず`dfp`の属性とメソッドに何があるかを確認する。

see(dfp)

主なものについて説明する。

属性`shape`は，`PanelData`の変数の数を表示する。以下が返り値の内容である。
```
(変数の数, 時間の観測値の数, 観察単位の数)
```

dfp.shape

* 変数の数：4（列にある変数）
* 時間の観測値の数：3（年）
* 観察単位の数：3（国）

メソッド`.mean()`を使うと、変数の観察単位毎の平均の`DataFrame`が返される。

dfp.mean()

引数に`time`を指定すると、変数の時間毎の平均が返される。

dfp.mean('time')

メソッド`demean()`は、変数の平均からの乖離が返される。即ち、変数$x$の平均が$\bar{x}$とすると、$x-\bar{x}$が返される。

dfp.demean()

`first_difference()`は変数の１階差分（$x_t-x_{t-1}$）が返される。

dfp.first_difference()

上の例では`NaN`があるため`Australia`と`UK`の行は１つしかない。

---
（注意）

`DataFrame`のメソッドは`PanelData`オブジェクトには使えない。

従って，`DataFrame`のメソッド（例えば，行や列の抽出）を使う場合，`DataFrame`に変換する必要がある。その場合，`PanelData`オブジェクトの属性`.dataframe`を使うことができる。

dfp.dataframe.loc['Japan',:]

### Balanced/Unbalancedの確認

データセットには欠損値がある場合がある。観察単位数が$N$で時間の観測値の数が$T$の場合，観測値の数は$n=N\times T$となるが，次の2つを区別する。
* balanced panel data：$n=N\times T$（観察単位に対して全ての期間の全ての変数に欠損値がない）
* unbalanced panel data：$n<N\times T$（欠損値がある）

balanced か unbalancedかは以下のコードで確認できる。まず，属性`isnull`を使う。

dfp.isnull

それぞれの行に`NaN`があれば`True`を、なければ`False`を返す。次に`True/False`を逆転させるために`~`を使う。

~dfp.isnull

`True`の行には`NaN`はなく、`False`の行に`NaN`がある。行数が多い場合はメソッド`all()`が便利である。`all()`は列に対して全ての要素が`True`の場合のみ`True`を返す。

(~dfp.isnull).all()

`False`なので unbalanced panel data ということが確認できた。

## １階差分推定（再考）

ここでは`linearmodels`を使い，以前行った１階差分推定を再考する。データ`crime4`を使う。

crime4 = wooldridge.data('crime4')
crime4.head()

`county`と`year`を使い`MultiIndex`化する。

crime4 = crime4.set_index(['county','year'])
crime4.head()

次に`PanelData`オブジェクトに変換しデータの特徴を調べる。

crime4p = PanelData(crime4)
crime4p.shape

* 57: 変数の数
* 7: 時間の観測値の数（年次データなので７年間）
* 90：観察単位の数（人数）

次にbalanced もしくは unbalanced data set かを確認する。

(~crime4p.isnull).all()

Unbalancedのデータセットだと確認できた。

---
実際に回帰式を書くことにする。使い方は`statsmodels`と似ている。
* `FirstDifferenceOLS`モジュールの関数`.from_formula`を使い次のように引数を指定する。

$$\text{.from_formula}(\text{回帰式}, \text{データ})$$

* 定数項を入れることはできない仕様となっている。
* ここでは，以前の推定結果と比べるために，ダミー変数`d82`を追加する。

formula = 'lcrmrte ~ d82 + d83 + d84 + d85 + d86 + d87 + lprbarr + \
                lprbconv + lprbpris + lavgsen + lpolpc'

１階差分モデルの設定（インスタンスの作成）

mod_dif = FirstDifferenceOLS.from_formula(formula, data=crime4)

`statsmodels`と同じように，そこから得た結果にメソッド`.fit()`を使い計算し結果が返される。

res_dif = mod_dif.fit()

＜結果の表示方法＞
1. `res_dif`もしくは`print(res_dif)`を実行。
1. `res_dif`には属性`summary`が用意されているが、表示方法1と同じ内容が表示される。
1. `summary`には属性`tables`があり，２つの表がリストとして格納されている。
    * `tables[0]`：検定統計量の表（`print()`を使うと見やすくなる）
    * `tables[1]`：係数の推定値やp値などの表（`print()`を使うと見やすくなる）

print(res_dif.summary.tables[1])

推定結果は以前のものと同じである。