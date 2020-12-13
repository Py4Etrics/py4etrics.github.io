# 質的変数と回帰分析

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import wooldridge

実証分析で使う変数を次の２種類に区別することができる。
1. 量的変数（体重，賃金など）
2. 質的変数（性別，人種，地域など）

今までは量的変数を考えたが，ここでは質的変数について議論する。まず男女のような２つの属性に分けるダミー変数を考える。その後に，より一般的な3つ以上の属性をに特徴付けられるカテゴリー変数を扱う場合を説明する。

## ダミー変数

### ケース１：定数項だけの回帰分析

#### ダミー変数なし

直感的な説明にするために説明変数が定数項だけの回帰分析から始める。
具体的には次の回帰式を考える。

$y=\beta_0+u$

実は，この場合のOLS推定量$\hat{\beta}_0$は被説明変数$y$の平均と等しいことになる。この結果を確認するために以下では`wooldridge`パッケージの`wage1`のデータを使う。

wage1 = wooldridge.data('wage1')
wooldridge.data('wage1',description=True)

時間平均賃金`wage`を被説明変数に設定する。`statsmodels`では，定数項だけの回帰式を考える場合，`1`をつか加える必要がある。

form_const = 'wage ~ 1'  # 定数項だけの場合は１が必要

res_const = ols(form_const, data=wage1).fit()

res_const.params

この推定値が賃金の平均と等しいこをを確認するために，直接平均を計算する。

wage1.loc[:,'wage'].mean()

この結果はGM仮定4の$\text{E}(u|X)=0$から簡単に導出できる。この仮定を標本回帰式で考えると次式となる。

$$\frac{1}{N}\sum_{i=1}^Nu_i=0$$

この式の左辺にに回帰式を代入すると

$$\frac{1}{N}\sum_{i=1}^N\left( y_i-\beta_0\right)=\bar{y}-\beta_0$$

この式が0となる$\hat{\beta_0}$がOLS推定量なので

$$\bar{y}=\hat{\beta_0}$$

となる。

#### ダミー変数あり：２つのケース

同じデータを使って$\{0,1\}$の値を取るダミー変数を考える。データセット`wage1`の中の`female`という変数があり，以下の値を取る。
```
女性の場合：female = 1
男性の場合：female = 0
```
値が０のカテゴリーを**基本カテゴリー**という。

$D$を`female`のダミー変数とすると回帰式は以下のようになる。

$$
y=\beta_0+\beta_1D
$$

さらに，この式は$D=\{0,1\}$の値によって以下のように表すことができる。

男性：$D=0\quad\Rightarrow\quad y=\beta_0+u$

女性：$D=1\quad\Rightarrow\quad y=\beta_0+\beta_1+u$

即ち，OLS推定量は以下を表すことになる。

$\hat{\beta}_0$：男性の平均賃金

$\hat{\beta}_0+\hat{\beta}_1$：女性の平均賃金

この回帰式を使い，時間賃金の男女間の差について考察する。

form_const_2 = 'wage ~ female'

res_const_2 = ols(form_const_2, data=wage1).fit()

res_const_2.params

* `female=0`の場合は男性なので，定数項の値（約7.10）が男性の時間賃金の平均である。
* `female=1`の場合は女性なので，女性の時間賃金の平均は

    $$7.10-2.51\approx 4.59$$

即ち，男性（基本カテゴリー）と比べて女性の平均賃金は2.51ドル低い。

データを使い直接男女の平均を計算して確認する。

＜女性の場合＞

# 女性だけを抽出するTrue/False条件の作成
cond_female = (wage1['female']==1)

wage1.loc[cond_female,'wage'].mean()

＜男性の場合＞

# 男性だけを抽出するTrue/False条件の作成
cond_male = (wage1['female']==0)

wage1.loc[cond_male,'wage'].mean()

**（解釈）**
* 女性の時間賃金は約2.51ドル低い
* しかし比較する場合，同じ条件で比べるべきではないだろうか。例えば，未婚・既婚，教育年数や就労期間が賃金に影響する場合，この差を考慮すべきである。しかし，この回帰式にはそのような変数は含まれていないため，含まれていない変数の違いが賃金の違いに反映されている可能性が高い。

#### ダミー変数あり：４つのケース

データセット`wage1`には`married`という変数が含まれており，以下の値をとる。
```
既婚者の場合：married = 1
未婚者の場合：married = 0
```
`female`と組み合わせることにより，次の４つのケースを分けることができる。
```
未婚男性：female=0, married=0
未婚女性：female=1, married=0
既婚女性：female=1, married=1
既婚男性：female=0, married=1
```
この４つのケースを捉えるために、`female`と`married`の値によって`0`もしくは`1`の値になるダミー変数を作成する。

---
＜ステップ１＞

新たなダミー変数の作成ルールを定義する関数を作成する。この関数では`DataFrame`の行を引数とする。

# 以下では row をDataFrameの行と考える。

# 未婚男性の関数
def singmale(row):
    if row['female'] == 0 and row['married'] == 0:
        return 1
    else:
        return 0

# 既婚男性の関数
def marmale(row):
    if row['female'] == 0 and row['married'] == 1:
        return 1
    else:
        return 0

# 未婚女性の関数
def singfem(row):
    if row['female'] == 1 and row['married'] == 0:
        return 1
    else:
        return 0

# 既婚女性の関数
def marfem(row):
    if row['female'] == 1 and row['married'] == 1:
        return 1
    else:
        return 0

＜ステップ２＞

上の関数を使い，`wage1`に新たな列を追加する。

以下のコードで使う`.apply()`は第１引数の関数を行または列に適用するメソッドである。`axis=1`は「任意の列に沿って各行に関数を適用する」ことを指定している（`axis='columns'`でもOK）。また、引数の関数に`()`は書かないことに注意しよう。即ち，引数に関数名（関数を参照する「参照記号」）を引数とすることにより，関数のオブジェクトを参照していることになる。即ち、`.apply()`は第１引数の関数を参照し、それを実行するメソッドである。

wage1.loc[:,'singmale'] = wage1.apply(singmale, axis=1)
wage1.loc[:,'marmale'] = wage1.apply(marmale, axis=1)
wage1.loc[:,'singfem'] = wage1.apply(singfem, axis='columns')
wage1.loc[:,'marfem'] = wage1.apply(marfem, axis='columns')

wage1.head(3)

一方でコードを書く際、似たものをコピペして少しだけ変更することを繰り返すとエラーや意図しない結果につながる可能性が高くなる。その場合は、次のように`for`ループで書くことを薦める。

まず辞書を作成する：
* キーを列のラベル
* 値を関数名

func_dict = {'singmale':singmale,
             'marmale':marmale,
             'singfem':singfem,
             'marfem':marfem}

次に`func_dict`を使い`for`ループで列を追加する。ここで`key`は辞書のキーを指す。

for key in func_dict:
    wage1.loc[:,key] = wage1.apply(func_dict[key], axis=1)

wage1.head(3)

---
これら４つのケースを捉えるために次の回帰式を使う。

$y=\beta_0+\beta_1D_1+\beta_2D_2+\beta_3D_3+u$

* 基本カテゴリー：`singmale`
* $D_1$=`marmale`
* $D_2$=`singfem`
* $D_3$=`marfem`

---
$D_1=\{0,1\}$、$D_2=\{0,1\}$、$D_3=\{0,1\}$の取る値を考慮すると，以下の４つのパターンに分けることができる。

$D_1=0$ & $D_2=0$ & $D_3=0\quad\Rightarrow\quad$
$y=\beta_0+u$

$D_1=1$ & $D_2=0$ & $D_3=0\quad\Rightarrow\quad$
$y=\beta_0+\beta_1+u$

$D_1=0$ & $D_2=1$ & $D_3=0\quad\Rightarrow\quad$
$y=\beta_0+\beta_2+u$

$D_1=0$ & $D_2=0$ & $D_3=1\quad\Rightarrow\quad$
$y=\beta_0+\beta_3+u$

即ち，OLS推定量は以下を表すことになる。

$\hat{\beta}_0$：未婚男性の平均賃金

$\hat{\beta}_0+\hat{\beta}_1$：既婚男性の平均賃金

$\hat{\beta}_0+\hat{\beta}_2$：未婚女性の平均賃金

$\hat{\beta}_0+\hat{\beta}_3$：既婚女性の平均賃金

form_const_4 = 'wage ~ marmale + singfem + marfem'

res_const_4 = ols(form_const_4, data=wage1).fit()

para4 = res_const_4.params
para4

（結果）
* 未婚男性の平均賃金は約5.16
* 未婚男性に比べて既婚男性の平均賃金は約2.82高い
* 未婚男性に比べて未婚女性の平均賃金は約0.56低い
* 未婚男性に比べて既婚女性の平均賃金は約0.60低い

`wage1`のデータから直接計算して確認する。

未婚男性の平均賃金

wage1.query('female==0 & married==0')['wage'].mean()

para4[0]

既婚男性の平均賃金

wage1.query('female==0 & married==1')['wage'].mean()

para4[0]+para4[1]

未婚女性の平均賃金

wage1.query('female==1 & married==0')['wage'].mean()

para4[0]+para4[2]

既婚女性の平均賃金

wage1.query('female==1 & married==1')['wage'].mean()

para4[0]+para4[3]

### ケース２：定量的変数の導入

１つのダミー変数`female`だけが入るケースに次の変数を加えた回帰式を考える。
* `educ`：教育年数
* `exper`：雇用経験年数
* `tenure`：勤続年数

form_1 = 'wage ~ female + educ + exper+ tenure'

res_1 = ols(form_1, data=wage1).fit()

res_1.params

（解釈）
* 賃金格差は約-1.81に減少した。これは`educ`, `exper`, `tenure`の影響を取り除いた結果である。言い換えると，教育，経験，就労期間を所与とすると（それらの変数が同じである場合という意味），女性の時間賃金は約1.8ドル低い。
* 女性差別を捉えているのだろうか？回帰式にない変数（その影響は誤差項に入っている）が残っている可能性があるので，必ずしもそうではない。またその場合，欠落変数バイアスも発生し得る。

### ケース３：ダミー変数の交差項

ケース１と２の被説明変数は`wage`をそのまま使ったが，ここでは対数を取り賃金方程式にダミー変数の交差項を入れて考察する。

以下の回帰式を考える。

$$y=\beta_0+\beta_1D+ \beta_2Dx+\beta_3x + u$$

ここで$D$がダミー変数，$x=$は定量的変数であり，$Dx$がダミー変数の交差項である。ダミー変数が取る値$D=\{0,1\}$に分けて考えると，以下を推定することになる。

$D=0\quad\Rightarrow\quad
y=\beta_0+\beta_3x + u$

$D=1\quad\Rightarrow\quad
y=\left(\beta_0+\beta_1\right)+ \left(\beta_2+\beta_3\right)x + u$



---
具体例として$D=$
`female`，$x=$`educ`とするとOLS推定量は以下を表すことになる。

$\hat{\beta}_0$：（教育の効果を取り除いた）男性の平均賃金（対数）

$\hat{\beta}_3$：男性の賃金に対する教育の効果（％）

$\hat{\beta}_0+\hat{\beta}_1$：（教育の効果を取り除いた）女性の平均賃金（対数）

$\hat{\beta}_2+\hat{\beta}_3$：女性の賃金に対する教育の効果（％）


（注意）

`statsmodels`の回帰式では$\text{female}\times \text{educ}$を$\text{female}:\text{educ}$と書く。

form_2 = 'np.log(wage) ~ female + female:educ + educ + exper + tenure'

res_2 = ols(form_2, data=wage1).fit()

print(res_2.summary().tables[1])

**$t$検定**
* `female`
    * 教育などの影響を省いた後の平均賃金の差
    * 5%有意水準で$\text{H}_0$`female`=0は棄却できない。
* `female:educ`

    * 教育などの影響を省いた後の教育の収益率の差
    * 5%有意水準で$\text{H}_0$
    `female:educ`=0は棄却できない。

**$F$検定**

$\text{H}_0$: `female`$=$
`female:educ`$=0$の制約を考えよう。

hypotheses = 'female=0, female:educ=0'

res_2.f_test(hypotheses).pvalue

$\text{H}_0$は棄却される。

$t$検定では，`female`と`female:educ`はそれぞれの帰無仮説が棄却されなかったが，$F$検定では制約が棄却された。一貫性がなく説明変数が不足している可能性がある。

## カテゴリー変数

カテゴリー変数とは定性的な変数であり，男女もカテゴリー変数の一種である。カテゴリー変数をダミー変数に変換するには2つの方法がある。

1. `statsmodels`にはカテゴリー変数に自動的にダミー変数を割り当てる機能がある。操作は簡単で，単に回帰式の中で`C()`の中にカテゴリー変数を入れるだけである。
1. `C()`を使わない方法。

### `C()`を使う方法

式の中にダミー変数を入れるためには，変数が`pandas`のカテゴリー型変数に変換される必要がある。回帰式の中で`C()`を使うと，文字型データなどは自動的に変換される。

#### 例１

例として，男女のカテゴリーがある`wage1`のデータセットをもう一度使う。

df = wage1.loc[:,['wage', 'female', 'educ']]

`df`のメソッド`replace()`を使って`female`の列の値を以下のルールに沿って変換し，それを`df`に`sex`という列として入れ直す。
* 1 $\Rightarrow$ `female`
* 0 $\Rightarrow$ `male`

`replace()`の中は辞書になっている（`replace()`の代わりに`map()`でもOK）。

df.loc[:,'sex'] = df['female'].replace({1:'female',0:'male'})

df.head(3)

`sex`の変数を`C()`に入れて回帰式を書いて計算。

form_c = 'wage ~  C(sex) + educ'

res_c = ols(form_c, data=df).fit()

res_c.params

---
`C(sex)[T.male]`について
* `T`は`Treatment`の頭文字で，通常のダミー変数を設定することを示している。
* `male`は`male`の変数であることを表しており，自動的に`female`が基本カテゴリーに設定されたことが分かる。

（結果）

`C(sex)[T.male]`は`female`に比べて`male`の賃金は約2.27ドル高いことを示している。

---
一方で，基本カテゴリーを手動で設定したい場合がある。その場合には`C()`に`Treatment("＜設定したいカテゴリー＞")`を引数として追加する。

（注意）`Treatment()`の中は double quotations `" "`を使用すること。

例えば，`male`を基本カテゴリーとする。

form_cm = 'wage ~  C(sex,Treatment("male")) + educ'

res_cm = ols(form_cm, data=df).fit()

res_cm.params

この結果は，`male`基本カテゴリーとする以下の結果と同じである。

form_ca = 'wage ~  female + educ'

res_ca = ols(form_ca, data=df).fit()

res_ca.params

#### 例２

米国ニュージャージーとペンシルベニアのファースト・フード店で人種と所得によって価格差別をおこなっているかを検証する例を取り上げる。`wooldridge`パッケージのデータセット`discrim`には以下の変数が含まれている。
* フライド・ポテトの平均価格（`pfries`）
* 人口における黒人の比率（`prpblck`）
* 平均所得（`income`）
* ４つのファースト・フード店（`chain`; カテゴリー変数で1~4の数字が使われている）
    * `1`: Berger King
    * `2`: KFC
    * `3`: Roy Rogers
    * `4`: Wendy's

discrim = wooldridge.data('discrim')
wooldridge.data('discrim',description=True)

それぞれのファースト・フード店の数を確認する。

discrim['chain'].value_counts()

OLS推定をおこなう。

form_p = 'np.log(pfries) ~ prpblck + np.log(income) + C(chain)'

res_p = ols(form_p, data=discrim).fit()

print(res_p.summary().tables[1])

自動的にBerger Kingが基本カテゴリーに設定されている。

（結果）
* BKとKFCの価格比較

    $$
    \ln P_{\text{BK}}-\ln P_{\text{KFC}}=-0.0682 \\
    \quad\Downarrow \\
    \dfrac{P_{\text{BK}}}{P_{\text{KFC}}}=e^{-0.0682}=1.071
    $$
    
* `prpblck`と`np.log(income)`の$p$値は`0`に近く，帰無仮説は１％有意水準でも棄却できる。

### `C()`を使わない方法

例２を使って説明する。まず説明の前準備として`discrim`の中で使う変数だけを取り出そう。

df_c = discrim.loc[:,['pfries', 'prpblck', 'income', 'chain']]
df_c.head()

`replace()`を使って`chain`の列の値を以下のルールに沿って変換する。
* `1` $\Rightarrow$ Berger King
* `2` $\Rightarrow$ KFC
* `3` $\Rightarrow$ Roy Rogers
* `4` $\Rightarrow$ Wendy's

df_c.loc[:,'chain'] = df_c['chain'].replace(
                        {1:'Berger_King',2:'KFC',3:'Roy_Rogers',4:'Wendys'})

df_c.head()

`DataFrame`の特徴を確認する。

df_c.info()

列`chain`は`object`となっており文字型であることがわかる。

＜`C()`を使わない方法＞
* `pandas`の関数である`pd.Categorical()`を使って列`chain`を`pandas`のカテゴリー型に変換する。

df_c['chain'] = pd.Categorical(df_c['chain'])
df_c.info()

列`chain`は`category`になっている。後は，`chain`をそのまま回帰式に使うだけである。

form_c = 'np.log(pfries) ~ prpblck + np.log(income) + chain'

res_c = ols(form_c, data=df_c).fit()

print(res_c.summary().tables[1])

`C()`を使って推定した結果と同じである。