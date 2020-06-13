# `linearmodels`

import pandas as pd
from linearmodels.panel.data import PanelData
from linearmodels.panel import FirstDifferenceOLS
import wooldridge

## 説明

`linearmodels`は`statsmodels`を補完する目的として開発されている。主に，パネルデータ，操作変数法を使った推定法やGMMを扱う場合には非常に重宝するパッケージである。しかし，`linearmodels`は`statsmodels`の両方を使う上で以下の点に注意する必要がある。
* 推定結果などのメソッドや属性が共通化されているわけではない。次の表に３つの例を挙げる。


|              | 推定結果の<br> 表を表示 | 残差を<br> 表示する<br> メソッド | 標準誤差を<br> 取得する<br> 属性 |
|-------------:|:-----------------------:|:--------------------------------:|:--------------------------------:|
| statsmodels  | .summary()              | .resid                           | .bse                             |
| linearmodels | .summary()              | .resids                          | .std_errors                      |


* `statsmodels`も`linearmodels`も回帰式を文字列で指定できるが，定数項の指定する方法が異なる。
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

まず，このトピックで使用するパッケージを導入する。

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

`MultiIndex`のまま要素・列・行の抽出およぼスライシングには様々な方法があり，複雑である。特に，スライシングをしたい場合，一番簡単なのは`reset_index()`で通常の`DataFrame`に戻し，スライシングし新たな`DataFrame`を作成するだけでも十分であろう。

以下では，`.loc[]`を使い`MultiIndex`のままでの抽出方法について簡単に説明する。その際，以下のルールは変わらない。

$$.\text{loc}\left[\text{行の指定},\text{列の指定}\right]$$

ただ，`行の指定`にリストやタプルを使うことになる（`列の指定`も同じ）。

他の方法については[このサイト](https://note.nkmk.me/python-pandas-multiindex-indexing/)と[このサイト](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)が参考になる。

#### １つの観察単位の抽出

１つの要素を抽出する場合は，タプルを使う。例えば，日本の2001年の`gdp`を抽出したい場合。

df.loc[('Japan',2001), 'gdp']

#### 行の抽出

上の例で列の指定を`:`にすると，指定した行に対して全ての列を抽出できる。

df.loc[('Japan',2001), :]

この場合，列に対してスライシングも可能。

df.loc[('Japan',2001), 'gdp':'con']

指定した行に対して個別に複数列を抽出したい場合は，タプルを使う。

df.loc[('Japan',2001), ('gdp','con')]

複数行の抽出にはリストで指定する。

df.loc[(['Japan','UK'],[2001,2002]), :]

#### 第０インデックスの観察単位の全て

第０インデックスにある，ある観察単位の全てのデータだけを抽出したい場合は，通常の`Pandas`の場合と同じ。

df.loc['Japan', :]

複数の場合。

df.loc[['Japan','India'], :]

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

`linearmodels`では`MultiIndex`化された`DataFrame`をそのまま読み込み推定することができる。一方で，`linearmodels`の関数`PanelData`を使い`MultiIndex`化された`DataFrame`を`PanelData`オブジェクトに変換すると分析に必要な計算を簡単にできるようになる。必須ではないが，知っていて損はしない関数である。

まず`df`を`PanelData`オブジェクトに変換する。

dfp = PanelData(df)
dfp

---
属性`shape`は，`PanelData`の変数の数を表示する。以下が返り値の内容である。

$$
\left(\text{変数の数},\text{期間数},\text{観察単位の数}\right)
$$

dfp.shape

* 変数の数：4（列にある変数）
* 期間数：3（年）
* 観察単位の数：3（国）

---
データセットには欠損値がある場合がある。観察単位数が$N$で期間数が$T$の場合，観測値の数は$n=N\times T$となるが，次の2つを区別する。
* balanced panel data：$n=N\times T$（観察単位に対して全ての期間の全ての変数に欠損値がない）
* unbalanced panel data：$n<N\times T$（欠損値がある）

balanced か unbalancedかは以下のコードで確認できる。まず，メソッド`count()`を使う。

dfp.count()

観察単位（国）に対して，それぞれの変数に欠損値ではない観測値がいくつ存在するかを`DataFrame`として返す。期間数は3なので，3より低い数字があれば欠損値の存在を表す。例えば，Australiaの`inv`には欠損値がある。

次のコードは，欠損値がある場合には`True`を返す。ここで`nobs`は期間数（この場合3）を返す属性である。

dfp.count() == dfp.nobs

`all()`は，列に対して全ての要素が`True`の場合のみ`True`を返すので，これを使い確認できる。

(dfp.count() == dfp.nobs).all()

`( )`はその中を先に評価する，という意味（数学と同じ）。変数が多い場合，`all()`を2回使うと全ての変数に対して評価するので便利である。

(dfp.count() == dfp.nobs).all().all()

`False`なので unbalanced panel data ということが確認できた。

---
変数の観察単位毎の平均の計算

dfp.mean()

---
変数の時間毎の平均の計算

dfp.mean('time')

---
変数の平均からの乖離　$x-\bar{x}$，$\bar{x}$は平均。

dfp.demean()

---
変数の１階階差の計算　$x_t-x_{t-1}$

dfp.first_difference()

---
（注意）

`DataFrame`のメソッドは`PanelData`オブジェクトには使えない。

従って，`DataFrame`のメソッド（例えば，行や列の抽出）を使う場合，`DataFrame`に変換する必要がある。その際，`PanelData`オブジェクトの属性`.dataframe`を使うことができる。

dfp.dataframe.loc['Japan',:]

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
* 7: 期間数（年）
* 90：観察単位の数（人数）

次に，balanced もしくは unbalanced data set かを確認する。

(crime4p.count()==crime4p.nobs).all().all()

---
実際に回帰式を書くことにする。使い方は`statsmodels`と似ている。
* `FirstDifferenceOLS`モジュールの関数`.from_formula`を使い次のように引数を指定する。

$$\text{.from_formula}(\text{回帰式}, \text{データ})$$

* 定数項を入れることはできない仕様となっている。
* ここでは，以前の推定結果と比べるために，ダミー変数`d82`を追加する。

formula = 'lcrmrte ~ d82 + d83 + d84 + d85 + d86 + d87 + lprbarr + \
                lprbconv + lprbpris + lavgsen + lpolpc'

* １階差分モデルのインスタンスの作成

mod_dif = FirstDifferenceOLS.from_formula(formula, data=crime4)

* `statsmodels`と同じように，そこから得た結果にメソッド`.fit()`を使い計算し結果が返される。

res_dif = mod_dif.fit()

＜結果の表示方法＞
1. `res_dif`を実行。
1. `res_dif`に関数`print()`を使うと見やすい。
1. `res_dif`には属性`summary`が用意されているが，表示方法1と同じ。
1. `summary`には属性`tables`があり，２つの表がリストとして格納されている。
    * `tables[0]`：検定統計量の表（`print()`を使うと見やすくなる）
    * `tables[1]`：係数の推定値やp値などの表（`print()`を使うと見やすくなる）

print(res_dif.summary.tables[1])

推定結果は以前のものと同じである。