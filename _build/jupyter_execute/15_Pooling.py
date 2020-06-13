# プーリング・データとパネル・データ

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import wooldridge
from scipy.stats import t

## 説明

独立に分布したプーリング・データ（independently distributed pooling data）とパネル・データ（panel data）を考える。両方とも横断面データと時系列データの両方の特徴を兼ね備えたデータである。

---
**独立に分布したプーリング・データ**には以下の特徴がある。
* ある期間（例えば，2000年から2010年の間）に期間毎無作為に観察単位（例えば，消費者や企業）が抽出される。従って，時間（年次データであれば年）が違えば観察単位は変化する。
* 時間の変化により必ずしも**同一分布**とはならない（即ち，分布は時間によって変化するかも知れない）。

日本の統計の例に[労働力調査](http://www.stat.go.jp/data/roudou/index.html)がある。

＜プーリング・データを使う利点＞
* 横断データと比べるとデータ数が増えるため，より正確な推定量や検定統計量を得ることが可能となる。
* 時間軸にそって独立した分布から標本を抽出するため，誤差項に自己相関はない。

---
**パネル・データ**には以下の特徴がある。
* 初期時点で無作為に観察単位（例えば，消費者や企業）を抽出し，同じ観測単位を時間軸に沿って観測して集めたデータである。
* 観測単位が同じであることが独立に分布したプーリング・データとの違いである。


＜パネル・データを使う意義＞

例として，各都道府県で行われた公共支出が県内の雇用に与える影響を推定したいとしよう。
* 観察単位：47都道府県 $i=1,2,...,47$
* 時系列範囲：2000~2020年 $t=2000,2001,...,2020$
* 変数：県内の雇用（$L$），公共支出（$G$），$x$は「その他の変数」

47都道府県と時間の２次元データとなっているため，次の推定方法が考えられる。
* 年別に横断面データとしてクロス・セクション分析行う。

    $$y_i = \beta_{0} + \beta_{1} G_i + \beta_{2}x_i + u_i$$

    しかし，それぞれの推定は20年間の間に起こる要因の変化の効果を無視することになり，公的支出の動的な側面を捉えていない。
* 47都道府県別に時系列の推定を行う。

    $$y_t = \beta_{0} + \beta_{1} G_t + \beta_{2}x_t+ u_t$$

    しかし，それぞれの推定は同じ日本の地域でありながら他の都道府県の影響を無視することになる。

このように2次元のデータの1つの軸に沿ったデータだけを使うと何らかの影響を無視することになりうる。換言すると，パネル・データを使い異なる観察単位の動的なデータから得る情報を有効に使うことにより，より正確な推定量や検定統計量を得ることができることになる。

## 独立に分布したプーリング・データ

このタイプのデータ扱う上で注意する点：
* 時系列の変数があるにも関わらず，被説明変数と少なくとも一部の説明変数とは時間不変の関係にあると暗に仮定することになる。
* 時系列の要素があるため，変数は必ずしも同一分布から抽出さる訳ではない。例えば，多くの経済で賃金と教育水準の分布は変化していると考える方が妥当である。この問題に対処するために時間ダミーを使うことになる。

`CPS78_85`のデータを使い推定方法を説明する。

# データの読み込み
cps = wooldridge.data('CPS78_85')

# データの内容の確認
wooldridge.data('CPS78_85', description=True)

このデータセットには1978年と1985年の変数があり，ダミー変数に次のものが含まれている。
* `y85`：1985年の時間ダミー変数
* `female`：男女のダミー変数
* `union`：労働組合員かどうかのダミー変数

これを使い，賃金と教育の関係を検証する。

# 回帰分析

formula = 'lwage ~ y85 + educ + female + \
                   y85:educ + y85:female + \
                   exper + I((exper**2)/100) + union'

result = ols(formula, cps).fit()

（コメント）
* `y85 + educ + female + y85:educ + y85:female`の部分は長いので`y85*(educ+female)`と省略して記述しても結果は同じ。
* `lwage`は名目賃金の対数の値である。この場合，実質賃金を考える方が自然なため，インフレの影響を取り除く必要がある。1978年と1985年の消費者物価指数をそれぞれ$p_{78}$と$p_{85}$とすると，その比率$P\equiv p_{85}/p_{78}$がその間のインフレの影響を捉える変数と考える。$P$を使い1985年の賃金を次式に従って実質化できる。 

    $$
    \ln\left(\text{実質賃金}_{85}\right) = \ln\left(\frac{\text{名目賃金}_{85}}{P}\right) = \ln\left(\text{名目賃金}_{85}\right)-\ln(P) = \text{85年のlwage}- \ln(P)
    $$

    この式から$\ln(P)$は回帰式の右辺の定数項に含まれることがわかる。即ち，上の式を変えることなく`educ`の係数は実質賃金で測った教育の収益率，`female`は実質賃金での男女賃金格差と解釈できる。

print(result.summary().tables[1])

教育の収益率
* `educ`の係数0.0747が1978年における教育1年の収益率（約7.5%）。統計的有意性は非常に高い（p-value=0)
* `y85:educ`の係数0.0185は1978年と1985年の値の差を表している。1985年の収益率は1.85%高く，5％水準で帰無仮説（1978年と1985年の収益率は同じ）を棄却できる。1985年の1年間の収益率は0.0747+0.0185=0.0932である（約9.3%）。

男女間の賃金格差
* `female`の係数`-0.3167`は1978年における値。即ち，女性の賃金は男性より約32%低く，統計的有意性は非常に高い（p値=0）。
* `y85:female`の係数0.0851は1978年と1985年の値の差を表しており，1985年の賃金格差は約8.5%改善したことを意味する。
* `y85:female`のp値0.098は両側検定であり，10％水準で帰無仮説（賃金格差は変化していない）を棄却できるが，5%水準では棄却できない。
* 一方，`y85:female`は正の値であり，女性の賃金の改善が統計的有意かどうかが問題になる。この点を考慮し片側検定をおこなう。
    * $H_0$: `y85:female`の係数 $＝0$
    * $H_a$: `y85:female`の係数 $>0$

t_value = result.tvalues['y85:female']  # y85:femaleのt値

dof = result.df_resid  # 自由度 n-k-1

`scipy.stats`を使い計算する。

1-t.cdf(t_value, dof)  # p値の計算

片側検定では，5%水準で帰無仮説を棄却できる。

## プーリング・データの応用：差分の差分析

### 説明

差分の差分析（Difference-in-Differences; DiD）を使うとプーリング・データを使い政策の効果を検証できる。

＜基本的な考え＞
* 例として，ゴミ焼却場の建築が近隣住宅価格に与える影響を考える。
* `t`=0：政策実施前の年
* `t`=1：政策実施後の年
* `D`：住宅がゴミ焼却場の近くであれば１，そうでなければ０（距離のダミー変数）
* `y`：住宅価格
* 仮定：ゴミ焼却場が建設されなかったとしたら，データにある全ての住宅価格は同じ率で増加していただろう。
* 次式を使って政策実施前と後それぞれで推定

    $$y_t = \alpha_t + \gamma_t D + u\qquad t=0,1$$
    
    * $\hat{y}_0^{\text{遠}}=\hat{\alpha}_0$：政策実施前の遠くに立地する住宅価格の平均
    * $\hat{y}_0^{\text{近}}=\hat{\alpha}_0+\hat{\gamma}_0$：政策実施前の近くに立地する住宅価格の平均
        * $\hat{\gamma}_0$：政策実施前の遠くと近くに立地する住宅価格の**差**
    * $\hat{y}_1^{\text{遠}}=\hat{\alpha}_1$：政策実施後の遠くに立地する住宅価格の平均
    * $\hat{y}_1^{\text{近}}=\hat{\alpha}_1+\hat{\gamma}_1$：政策実施後の近くに立地する住宅価格の平均
        * $\hat{\gamma}_1$：政策実施後の遠くと近くに立地する住宅価格の**差**
* 解釈（下の図を参考に）：
    * $\hat{\gamma}_0$は政策実施前の住宅価格の差を示す。
    * $\hat{\gamma}_1$は政策実施後の住宅価格の差を示す。
    * $\left(\hat{\gamma}_1-\hat{\gamma}_0\right)$は政策実施後と前の住宅価格の差の差を示す。この「差の差」の変化で近くに立地する住宅価格の変化を考えることができる。もしこの差の差が`0`であれば（即ち，差は同じだった），住宅価格は影響を受けなかったと解釈できる。もしこの差がマイナスであれば（即ち，差に違いが生じた）近くに立地する住宅の価格は減少してしたと考えられる。

```{image} /image/did.png
:scale: 30%
:align: center
```

上の議論から次のことが分かる。
* 住宅価格の変化は上の図で示しているように次の「差の差」で捉えることができる。

    $$
    \begin{align*}
    \hat{\gamma}_1^{\text{}}-\hat{\gamma}_0&=\hat{y}_1^{\text{近}}-\hat{y}_0^{\text{近}}-\left(\hat{\alpha}_1-\hat{\alpha}_0\right) \\
    &=\left(\hat{y}_1^{\text{近}}-\hat{y}_0^{\text{近}}\right)-\left(\hat{y}_1^{\text{遠}}-\hat{y}_0^{\text{遠}}\right) \\ 
    &=\left(\hat{y}_1^{\text{近}}-\hat{y}_1^{\text{遠}}\right)-\left(\hat{y}_0^{\text{近}}-\hat{y}_0^{\text{遠}}\right)
    \end{align*}
    $$

* 住宅価格の変化の間違った計算：
    * $\hat{y}_1^{\text{近}}-\hat{y}_0^{\text{近}}=\left(\hat{\alpha}_1-\hat{\alpha}_0\right)+\left(\hat{\gamma}_1-\hat{\gamma}_0\right)$
    * $\left(\hat{\alpha}_1-\hat{\alpha}_0\right)$は，ゴミ焼却場建設から影響を「受けない（仮定）」遠
    い場所に立地する住宅価格が時間と共に変化する部分を捉えており，ゴミ焼却場建設と関係なく発生する価格変化である。この部分を取り除かなければ，ゴミ焼却場建設の効果の推定量にバイアスが発生する。

---
＜`DiD`推定方法＞
* 通常の`OLS`を使うが，時間ダミー変数を導入する。
    * `T`：`t`=0の場合は０，`t`=1の場合は１
* 推定式：

    $$
    y=\beta_0+\delta_0T + \beta_1D + \delta_1TD + u
    $$
    

＜ダミー変数の値により4つのケースがある＞
1. `T`=`D`=0の場合

    $$
    y=\beta_0 + u
    $$

    $\hat{y}_{0}^{\text{遠}}\equiv\hat{\beta}_0$：政策実施前の遠くに立地する住宅の平均価格
    <br>

1. `T`=0 と `D`=1の場合

    $$
    y=\beta_0 + \beta_1D + u
    $$

    $\hat{y}_{0}^{\text{近}}\equiv\hat{\beta}_0+\hat{\beta}_1$：政策実施前の近くに立地する住宅の平均価格
    <br>
    
1. `T`=1 と `D`=0の場合

    $$
    y=\beta_0 + \delta_0T + u
    $$

    $\hat{y}_{1}^{\text{遠}}\equiv\hat{\beta}_0+\hat{\delta}_0$：政策実施後の遠くに立地する住宅の平均価格
    <br>
    
1. `T`=`D`=1の場合

    $$
    y=\beta_0 + \delta_0T + \beta_1D + \delta_1TD + u
    $$

   $\hat{y}_{1}^{\text{近}}\equiv\hat{\beta}_0+\hat{\delta}_0+\hat{\beta}_1+\hat{\delta}_1$：政策実施後の近くに立地する住宅の平均価格

ここでの関心事は$\hat{\delta}_1$の推定値（負かどうか）であり，その統計的優位性である。この定義を使うと次式が成立することが確認できる。

$$\hat{\delta}_1=\left(\hat{y}_1^{\text{近}}-\hat{y}_1^{\text{遠}}\right)-\left(\hat{y}_0^{\text{近}}-\hat{y}_0^{\text{遠}}\right)$$

これは上で導出した式にある$\hat{\gamma}_1^{\text{}}-\hat{\gamma}_0$と同じである。

### `DiD`推定

`keilmc`のデータを使いゴミ焼却場建設の近隣住宅価格に対する効果を実際に推計する。

kielmc = wooldridge.data('kielmc')

wooldridge.data('kielmc', description=True)

formula = 'rprice ~ nearinc + y81 + nearinc:y81'

result = ols(formula, data=kielmc).fit()

（コメント）
* 右辺を`nearinc * y81`と省略可能。

print(result.summary().tables[1])

* `nearinc:y81`の係数は約-11860であり，10%水準の両側検定でも帰無仮説（係数は0）を棄却できない。
* `nearinc:y81`は負の値であり住宅価格が減少したかを検証したい。従って，次の片側検定を行う。
    * $H_0$: `nearinc:y81`の係数 $＝0$
    * $H_a$: `nearinc:y81`の係数 $<0$

t_value = result.tvalues['nearinc:y81']  # t値

dof = result.df_resid  # 自由度 n-k-1

t.cdf(t_value, dof)  # p値の計算

片側検定では，10%水準で帰無仮説は棄却できるが，5%水準では棄却できない。負の効果のある程度の統計的有意性はあるようである。一方，上の回帰分析には次のことが言える。
* 左辺には住宅価格の対数を置く方が妥当ではないか。それにより，価格変化をパーセンテージ推計できると同時に，実質価格という解釈も成立する。
* 住宅価格に与える他の変数が欠落している可能性がある。

---
この2点を踏まえて，再度推定をおこなう。

まず，`NumPy`を使い住宅価格に対数をとり推計する。

formula_1 = 'np.log(rprice) ~ nearinc * y81'

result_1 = ols(formula_1, data=kielmc).fit()

print(result_1.summary().tables[1])

（結果）
* 特に`nearinc:y81`の係数の統計的有意性が上昇したわけではない。

次に，住宅価格に影響を及ぼしそうな変数を追加する。

formula_2 = 'np.log(rprice) ~ nearinc * y81 + age + I(age**2) + \
            np.log(intst) + np.log(land) + np.log(area) + rooms + baths'

result_2 = ols(formula_2, data=kielmc).fit()

print(result_2.summary().tables[1])

（結果）
* `nearinc:y81`の係数は，住宅価格が13.1%下落したことを示している。
* p値はこの経済的効果が非常に有意にゼロではないことを示している。

## パネル・データとその問題点

### パネル・データについて

パネル・データを使い推定する場合，観察単位$i$と時間$t$の２つ次元を考慮した回帰式となる。説明を具体化するために，47都道府県での雇用に対する公共支出の効果を考えよう。

$$
y_{it}= \alpha_{it} +\beta_1x_{it}+u_{it}\qquad i=1,2,...,n\quad t=1,2
$$

ここで

$$
\alpha_{i}=\beta_0+a_{i}+\delta_0D_t
$$

* $y_{it}$：都道府県$i$の$t$時点における雇用
* $x_{it}$：都道府県$i$の$t$時点における公共支出
* $\alpha_{it}$：回帰曲線の切片。
    * $i$がある（都道府県によって切片が異なる）理由：都道府県の間に存在しうる異質性を捉えている。
        * 例えば，他県の公共支出からの影響を受けやすい（若しくは，受けにくい）県がある可能性がある。また，海に接している県や内陸の県では公共支出の効果は異なるかも知れない。また働き方や公共支出に対する県民特有の考え方や好みの違いもありうる。こういう要素は変化には時間が掛かり，推定期間内であれば一定と考えることができる。ありとあらゆる異質性が含まれるため，観察不可能と考えられる。
    * $t$がある理由：公共支出以外の理由で時間と共に雇用は変化するかもしれない。時間トレンドの効果を捉える。
    * 都道府県特有の定数項は3つに分ける。
        * $\beta_0$：共通の定数項
        * $a_{i}$：都道府県別の定数項（**観察不可能**）
        * $\delta_0D_t$：時間による切片の変化を捉えており，$D_t$は時間ダミー変数
            * $D_0=0$
            * $D_1=1$
* $u_{it}$：時間によって変化する観測不可能な誤差項（$i$と$t$に依存する誤差項であり，idiosyncratic errorsと呼ばれることがある）
    * 仮定：$\text{Cov}(x_{it},u_{it})=0$

### OLS推定の問題点

$a_i$は観察不可能なため，上の回帰式は次のように表すことができる。

$$
y_{it}= \beta_0 + \delta_0D_t + \beta_1x_{it}+e_{it}\qquad e_{it}=a_i+u_{it}
$$

ここで$e_{it}$は観察不可能な誤差項。
* $x_{it}$と$a_i$とが相関する場合，$x_{it}$と$e_{it}$は相関することになり，GM仮定４は成立しなくなる。

    $$
    \text{Cov}\left(x_{it}a_{it}\right)\neq 0\quad \Rightarrow\quad \text{Cov}\left(x_{it}e_{it}\right)\neq 0
    $$

  即ち，$\hat{\beta}_1$は不偏性と一致性を失うことになる。
* 「異質性バイアス」と呼ばれるが，本質的には$a_i$の欠落変数バイアスである。

## １階差分推定の準備：`groupby`

パネル・データを使った推定方法として１階差分推定（First Differenced Estimator）があり，それを使い推定する。その前準備として変数の差分の作成方法について説明する。

`DataFrame`にはメソッド`groupby`があり，これを使うとカテゴリー変数がある列に従って行をグループ分けしグループ毎の様々な計算が可能となる。具体的な例としては[Gapminder](https://github.com/Haruyama-KobeU/Py4Etrics/blob/master/Gapminder.ipynb)を参照。ここでは`groupby`を使い差分の変数の作り方を説明する。

例として使うデータを読み込む。

# url の設定
url = 'https://raw.githubusercontent.com/Haruyama-KobeU/Haruyama-KobeU.github.io/master/data/data4.csv'

# 読み込み
df = pd.read_csv(url)

df

列`country`がカテゴリー変数であり，3つの国がある。この列を使いグループ分けする。

まず前準備として，メソッド`.sort_values()`を使い列`country`で昇順に並び替え，その後，列`year`で昇順に並び替える。

（コメント）

以下のコードには`.reset_index(drop=True)`が追加されているが，これは行のインデックスを0,1,2,..と振り直すメソッドである。引数`drop=True`がなければ，元々あったインデックスが新たな列として追加される。試しに，最初から`df`を読み直して`.reset_index(drop=True)`を省いて下のコードを実行してみよう。

df = df.sort_values(['country','year']).reset_index(drop=True)

df

`country`でグループ化した`df_group`を作成する。

df_group = df.groupby('country')

df_group

このコードの出力が示すように，`df_group`は`DataFrameGroupBy`というクラス名のオブジェクト（データ型）であり，`DataFrame`とは異なる。これによりグループ別の計算が格段と容易になる。

例えば，次のコードを使いそれぞれの経済の`gdp`の平均を計算することができる。

df_group['gdp'].mean()

`mean()`は`df_group`に備わるメソッドである。他にも便利な使い方があるが，ここでは変数の差分の作りかを説明する。

`df_group`にある変数の差分をとるためにメソッド`diff()`を使う。

var = ['gdp', 'inv']

df_diff = df_group[var].diff()

df_diff

次に`df`と`df_diff`を横に結合する。

**＜方法１：`pd.concat`＞**
1. `df_diff`の列名を変更
1. `pd.condat`を使い結合

df_diff_1 = df_diff.rename(columns={'gdp':'gdp_diff','inv':'inv_diff'})

df_diff_1

pd.concat([df, df_diff_1], axis='columns')

上のコードの`axis='columns'`は`axis=1`としてもOK。この方法は、`df`と`df_diff_1`の行の順序が同じという想定のもとで行っている。行の順番が同じでない場合や不明な場合は、次の方法が良いであろう。

**＜方法２：`pd.merge`＞**

`df`と`df_diff`を使い、方法１の２つのステップを１行で済ませる。次のコードでは3つの引数を指定している。
* `left_index=True`
    * `df`の行のインデックスに合わせて結合する。
* `right_index=True`
    * `df_diff`の行のインデックスに合わせて結合する。
* `suffixes=('', '_diff')`
    * 左の引数`''`：結合後，重複する左の`DataFrame`の列につける接尾辞（空の接尾辞）
    * 右の引数`'_diff'`：結合後，重複する右の`DataFrame`の列につける接尾辞

pd.merge(df, df_diff, left_index=True, right_index=True, suffixes=('', '_diff'))

`df`を上書きしたい場合は，`df=`を付け加える。

## １階差分推定

### 説明

異質性バイアスの一番簡単な解決方法が１階差分推定（First Differenced Estimator）である。考え方は非常に簡単である。次の１階差分を定義する。
* $\Delta y_{i}=y_{i1}-y_{i0}$
* $\Delta D = D_{1}-D_0 =1-0= 1$
* $\Delta x_{i}=x_{i1}-x_{i0}$
* $\Delta e_{i}=e_{i1}-e_{i0}=a_i+u_{i1}-\left(a_i+u_{i0}\right)=\Delta u_{i}$
    * $a_i$が削除される。

これらを使い，上の式の１階差分をとると次式を得る。

$$
\Delta y_{it}= \delta_0 + \beta_1\Delta x_{i}+\Delta u_{i}\qquad i=1,2,...,n
$$

* 良い点
    * （仮定により）$\Delta x_{i}$と$\Delta u_{i}$は相関しないため，GM仮定４は満たされることになる。
* 悪い点
    * $t=0$のデータを使えなくなるため，標本の大きさが小さくなる。
    * 期間内の説明変数$\Delta x_i$の変化が小さいと，不正確な推定になる（極端な場合，変化がないと推定不可能となる）。

---
＜推定方法＞

データから１階差分を計算し，`statsmodels`を使いOLS推定すれば良い。以下ではパッケージ`wooldridge`にあるデータを使う。

### 推定：２期間の場合

* 以下では`crime2`のデータを使い説明する。
* このデータを使い，失業率が犯罪に負の影響を与えるかどうかを検証する。

データの読み込み。

crime2 = wooldridge.data('crime2')

変数の説明。

wooldridge.data('crime2', description=True)

興味がある変数
* `crmes`：犯罪率（1000人当たり犯罪数）
* `unem`：失業率

（コメント）

データセットにはこの2つの変数の差分（`cunem`と`ccrmrte`）も用意されているが，以下では`groupby`を使って変数を作成する。

crime2.head()

このデータセットには，観察単位識別用の変数がない。しかし，行0と1が1つの地域のデータ，行2と3が別の地域のデータをなっていることがわかる（`year`を見るとわかる）。まず観察単位識別用のの列を作成する。

# 観察単位の数
n = len(crime2)/2

# 観察単位ID用のリスト作成 [1,2,3,4....]
lst = [i for i in range(1,int(n)+1)]

# 観察単位ID用のリスト作成 [1,1,2,2,3,3,4,4....]
country_list = pd.Series(lst).repeat(2).to_list()

`country_list`の説明：
1. `Series`のメソッド`repeat(2)`は`lst`の要素を２回リピートする`Series`を生成する。
1. `to_list()`は`Series`をリストに変換するメソッド。

データセットに列`county`を追加し確認する。

crime2['county'] = country_list  # 追加

crime2.loc[:,['county','year']].head()  # 確認

`unem`と`crmrte`の差分を求める。

var = ['unem', 'crmrte']  # groupbyで差分を取る列の指定

names = {'unem':'unem_diff', 'crmrte':'crmrte_diff'}  # 差分の列のラベル

crime2_diff = crime2.groupby('county')[var].diff().rename(columns=names)

crime2_diff.head()

`crime2_diff`を使って回帰分析をおこなう。

（コメント）

以下の計算では`NaN`は自動的に無視される。

formula_1 = 'crmrte_diff ~ unem_diff'

result_1 = ols(formula_1, crime2_diff).fit()

print(result_1.summary().tables[1])

＜結果＞
* 失業率が1%増加すると，犯罪率は約2.2%上昇する。
    * この場合、微分を計算するのではなく、`unem_diff`に`1`を代入して解釈する。
* 5％の水準で帰無仮説を棄却できる。
* 次の片側検定を行う
    * $H_0$: `unem`の係数 $＝0$
    * $H_a$: `unem`の係数 $>0$

t_value = result_1.tvalues['unem_diff']  # t値

dof = result_1.df_resid  # 自由度 n-k-1

1-t.cdf(t_value, dof)  # p値の計算

片側検定では，1%水準で帰無仮説を棄却できる。

---
**＜参考１＞**

１階差分推定を使わずに直接OLS推定するとどうなるかを確かめてみる。

formula_ols_1 = 'crmrte ~ d87 + unem'

result_ols_1 = ols(formula_ols_1, crime2).fit()

print(result_ols_1.summary().tables[1])

* 失業の効果は過小評価されている。
    * 地域の異質性を考慮しなため異質性バイアス（欠落変数バイアス）が発生している。
* 統計的有意性も低い（p値は非常に高い）。

---
**＜参考２＞**

参考１の回帰式にはダミー変数`d87`が入っており、年によって失業と犯罪の関係は異なることを捉えている。もしこのダミーが入らないと通常考えられる関係と逆の相関が発生する。

formula_ols_2 = 'crmrte ~ unem'

result_ols_2 = ols(formula_ols_2, crime2).fit()

print(result_ols_2.summary().tables[1])

### 推定：３期間以上の場合

`crime4`のデータセットを使い，犯罪の決定要因について推定する。

crime4 = wooldridge.data('crime4')
crime4.head()

wooldridge.data('crime4', description=True)

＜興味がある変数＞
* 被説明変数
    * `lcrmrte`：１人当たり犯罪数（対数）
* 説明変数
    * `lprbarr`：逮捕の確率（対数）
    * `lprbconv`：有罪判決の確率（対数; 逮捕を所与として）
    * `lprbpris`：刑務所に収監される確率（対数; 有罪判決を所与として）
    * `lavgsen`：平均服役期間（対数）
    * `lpolpc`：１人当たり警官数（対数）

それぞれの変数の差分も用意されているが，以下では列`county`をグループ化しそれらの値を計算する。

# グループ化
crime4_group = crime4.groupby('county')

# 差分を計算したい変数
var = ['lcrmrte', 'lprbarr', 'lprbconv', 'lprbpris', 'lavgsen', 'lpolpc']

# 差分のDataFrame
crime4_diff = crime4_group[var].diff()

# DataFrameの結合
crime4 = pd.merge(crime4, crime4_diff, 
                  left_index=True, right_index=True,
                  suffixes=('','_diff'))

---
推定式については，2期間モデルと同じように考える。違う点は，7年間の年次データであるため，６つの時間ダミー変数を入れることだけである。

formula_2 = 'lcrmrte_diff ~ d83 + d84 + d85 + d86 + d87 + \
                            lprbarr_diff + lprbconv_diff + \
                            lprbpris_diff + lavgsen_diff + \
                            lpolpc_diff'

result_2 = ols(formula_2, crime4).fit()

print(result_2.summary().tables[1])

＜結果＞
* `lprbarr`（逮捕の確率），`lprbconv`（有罪判決の確率），`lprbpris`（収監確率），`lavgsen`（平均服役期間）は全て予想通りの結果。
* しかし平均服役期間はの統計的優位性は低い。（犯罪予防効果は低い？）
* `lpolpc`（１人当たり警官数）の効果は正
    * 警官が多くなると，犯罪数が同じであっても犯罪の報告は増える？
    * 同時性バイアス？