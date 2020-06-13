# 統計学の簡単な復習

## 確率変数

**定義**

確率変数（random variables）とは，無作為のプロセスの結果として実数をとる変数であり，実現し観察されるまで値が未知の変数である。実現した値を実現値もしくは観察値と呼ぶ。次の記号を使って例を考える。
* $X$：確率変数自体を示す記号
* $x$：実現値

例１（離散型確率変数）
* サイコロの目：$X$
* 実現可能な値の集合：$x\in\{1, 2, 3, 4, 5, 6\}$
* 実現値：$x=3$

例２（連続型確率変数）
* ランダムに選んだ経済学部の学生の身長：$X$
* 実現可能な値の集合：$\{x\;|\;0< x<\infty\}$
* 実現値：175.920483.....cm

---
**確率変数のある値が発生する確率**
* 例１：サイコロ
    * $X=3$の確率は確率質量関数 $f(3)=1/6$で表される。
    * 実現可能な値の確率の合計＝１，即ち $\displaystyle\sum_{x=1}^6f(x)=1$
* 例２：ランダムに選んだ経済学部の学生の身長
    * $X=175.92$の確率は確率密度関数 $f(175.92)=0.0204$で表される。
    * 実現可能な値の確率の合計＝１，即ち $\displaystyle\int_0^{\infty}f(x)dx=1$

## 確率変数の特徴を示す尺度

（以下で使う記号）

$X,Y$：確率変数

$x,y$：確率変数の実現値

---
**期待値（expected value）＝ 平均（average or mean）**

$\text{E}(X)=\mu_X$

（性質）
* $\text{E}(X)\gtreqqless 0$
* $\text{E}(aX)=a\text{E}(X)$
* $\text{E}(X+Y)=\text{E}(X)+\text{E}(Y)$
* $\text{E}(XY)=\text{E}(X)\cdot\text{E}(Y)+\text{Cov}(X,Y)$
* $X$の単位に依存

---
**分散（variance）**

$\sigma_X^2\equiv\text{Var}(X)\equiv\text{E}\left[(X-\mu_X)^2\right]=\text{E}\left[X^2\right]-\mu_X^2$

（性質）
* $\text{Var}(X)\geq 0$
* $\text{Var}(X+a)=\text{Var}(X)$
* $\text{Var}(aX)=a^2\text{Var}(X)$
* $\text{Var}(aX+bY)=a^2\text{Var}(X)+b^2\text{Var}(Y)+2ab\cdot\text{Cov}(X,Y)$
* $\text{Var}(aX-bY)=a^2\text{Var}(X)+b^2\text{Var}(Y)-2ab\cdot\text{Cov}(X,Y)$
* $X$の単位に依存

---
**標準偏差（standard deviation）**

$\sigma_X\equiv\sqrt{\text{Var}(X)}$

（性質）
* $X$の単位に依存

---
**共分散（covariance）**

$\sigma_{XY}\equiv\text{Cov}(X,Y)=\text{E}\left[(X-\mu_X)(Y-\mu_Y)\right]$

（性質）
* $\text{Cov}(X,Y)\lesseqqgtr 0$
* $\text{Cov}(X,X)=\text{Var}(X)$
* $X$と$Y$の単位に依存

---
**相関係数（correlation coefficient）**

$\rho_{XY}\equiv\text{Corr}(X,Y)=\dfrac{\sigma_{XY}}{\sigma_X\cdot\sigma_Y}
=\dfrac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\cdot\text{Var}(Y)}}$

（性質）
* $-1\leq\rho_{XY}\leq 1$
* $X$と$Y$の単位に依存しない

## 正規分布（Normal Distribution）

別名：ガウス分布（Gausian Distribution）

### 確率密度関数と累積分布関数

**確率密度関数**

$$
\phi(x)=\dfrac{1}{\sqrt{2\pi\sigma_X}}e^{-\frac{1}{2}\left(\frac{x-\mu_X}{\sigma_X}\right)^2}
$$

* ２つのパラメータ：平均（$\mu_X$）と分散（$\sigma_X^2$）
* 左右対称
* 「$X$は平均$\mu_X$，分散$\sigma_X^2$の正規分布に従う」を記号で表現

$$X\sim N\left(\mu_X,\sigma_X^2\right)$$

**確率分布関数**

$$F(x)=\int_{-\infty}^x\phi(s)ds$$

### 標準正規分布

正規分布の変数$X$を次式

$$
Z=\dfrac{X-\mu_X}{\sigma_X}
$$

で変換すると，$Z$の分布は標準正規分布に従う。

$$Z\sim N(0,1)$$

### 多変量正規分布

* ２つの確率変数$X_1$と$X_2$を考えよう。

    $$
    X_1\sim N\left(\mu_1,\sigma_1^2\right),\qquad\qquad
    X_2\sim N\left(\mu_2,\sigma_2^2\right)
    $$
    
    * この表記からは２つの確率変数に何らかの関係性が存在するかどうか不明であるが，通常この場合，$X_1$と$X_2$は「独立」と受け取られる。
    
* この２つの変数に「何らかの関係性」を明確にするために，次のようにまとめて書く：

    $$
    \begin{bmatrix}
        X_1\\X_2
    \end{bmatrix}
    \sim
    N\left(
    \begin{bmatrix}
        \mu_1\\ \mu_2
    \end{bmatrix}
    ,
    \begin{bmatrix}
        \sigma_1^2,&\sigma_{12}\\
        \sigma_{21},& \sigma_2^2    
    \end{bmatrix}
    \right)
    $$

    もしくは

    $$
    X\sim N\left(\mu_X,\Sigma_X\right)
    $$

    * $X$：確率変数のベクトル $\left(X_1,X_2\right)^T$（$T$は「置換する」という意味で，列ベクトルにしている）
    * $\mu_X$：平均のベクトル $\left(\mu_1,\mu_2\right)^T$
    * $\Sigma_X$：分散共分散行列
    
        $$
        \Sigma_X=
        \begin{pmatrix}
            \sigma_1^2,&\sigma_{12}\\
            \sigma_{21},& \sigma_2^2
        \end{pmatrix}
        $$
        
         * $\sigma_{12}=\sigma_{21}$は$X_1$と$X_2$の共分散
         * 上で「何らかの関連性」と書いたが，それを$\sigma_{12}$が捉えている。

（共分散の解釈）
* $\sigma_{12}=0$：$X_1$と$X_2$は独立であり何の関係もない。即ち，
    $X_1\sim N\left(\mu_1,\sigma_1^2\right),\;X_2\sim N\left(\mu_2,\sigma_2^2\right)$
  と別々に書いて何の問題もない。
* $\sigma_{12}>0$：$X_1$と$X_2$は「同じ方向」の値が抽出される傾向にある。例えば，両辺数ともプラスの値，もしくはマイナスの値。$\sigma_{12}$が大きくなれば，その傾向はより強くなる。（注意：これは傾向であり，必ずそうはならない）
* $\sigma_{12}<0$：$X_1$と$X_2$は「逆方向」の値が抽出される傾向にある。例えば，$X_1$はプラスの値で$X_2$はマイナスの値，もしくはその反対。$\sigma_{12}$の絶対値が大きくなれば，その傾向はより強くなる。（注意：これは傾向であり，必ずそうはならない）

## 標本の特徴を示す数値的尺度

母集団から標本を無作為に１つデータを抽出するとしよう。その場合，
* 母集団 ＝ 実現可能な値の集合
* 抽出するデータ ＝ 確率変数
* 抽出後の値 ＝ 実現値

この場合，標本の大きさは母集団より小さい（母集団の大きさが2以上と仮定）。

上では１つのデータだけを抽出を考えたが，通常実証分析では複数のデータを扱い，データの種類によって母集団の大きさと標本の大きさを以下のように解釈することが可能である。

* 時系列データ
    * 時間は無限に続くため，無限の母集団からの標本抽出 $\Rightarrow$ 標本の大きさは母集団より小さい。
* 横断面データ
    * 多くの場合，費用対効果から母集団から一部を標本を収集する $\Rightarrow$ 標本の大きさは母集団より小さい。
    * 母集団の大きさが小さい場合，標本の大きさは母集団の大きさと「等しい」ケースがある。
        * 例えば，2018年神戸大学経済学部の中級マクロ経済学I定期試験の点数の場合，約300のデータ。
        * この場合でも，標本の大きさは母集団より小さいと考えることができる。
            * ある学生$i$さんの点数は確率変数と解釈できる。その場合，実現可能な値の集合（小数点は無視）は
                $\left\{0,1,2,3,....,97,98,99,100\right\}$であり，点数の種類は101ある。この中なら１つの値だけが実現値として観察されている。更に，約300名の学生が試験を受けたので，母集団の大きさは約$101\times 300=20200$となる。

（「標本の大きさは母集団より小さい」の含意）
* 母集団のパラメータを推定するための標本の統計量には必ず**誤差**が存在する。

（コメント）

標本のそれぞれの観測値が，同じ母集団から独立に（他の観測値との何の関連性もなく）抽出された場合，それらは
**独立同一分布（idependently identically distributed;　略して IID）**に従うと呼ぶ。

---
（以下で使う記号）
* 標本の大きさ：$n$
* $i$番目の確率変数：$X_i$

---
**標本平均（sample mean）**
* 確率変数の標本平均：$\bar{X}=\dfrac{1}{n}\displaystyle\sum_{i=1}^nX_i$
* 標本平均の実現値：$\bar{x}=\dfrac{1}{n}\displaystyle\sum_{i=1}^nx_i$

（特徴）
* $\bar{X}$は母集団平均の不偏推定量

    $$\text{E}(\bar{X})=\mu_X$$
    
* $\bar{x}$はその推定値。
* $X_i$がIIDの場合の$\bar{X}$の分散

    $$
    \text{Var}(\bar{X})=\dfrac{1}{n}\sigma_{X}^2
    $$
    
    * $n\rightarrow\infty\;\Rightarrow\;\text{Var}(\bar{X})=0$

---
**標本分散（sample variance）**
* 確率変数の標本分散：$\hat{\sigma}_X^2=\dfrac{1}{n-1}\displaystyle\sum_{i=1}^n\left(X_i-\bar{X}\right)^2$
* 標本分散の実現値：$\hat{\sigma}_x^2=\dfrac{1}{n-1}\displaystyle\sum_{i=1}^n\left(x_i-\bar{x}\right)^2$

（特徴）
* $\hat{\sigma}_X^2$は母集団分散の不偏推定量
* $\hat{\sigma}_x^2$はその推定値

（注意）
* 分母は $n-1$であり，これにより$\hat{\sigma}_X$は母集団分散の不偏推定量となる。

---
**標本平均の分散**
* 確率変数の標本平均の分散$\text{Var}(\bar{X})=\dfrac{1}{n}\sigma_X^2$にある$\sigma_X^2$は母集団の分散であり観測不可能。従って，推定する必要がある。その推定量として$\hat{\sigma}_X$を使う。

    $$
    \widehat{\text{Var}(\bar{X})}=\frac{1}{n}\hat{\sigma}_X^2
    $$
    
* 以下を標準誤差と呼ぶ

    $$
    \text{SE}(\bar{X})=\sqrt{\widehat{\text{Var}(\bar{X})}}=\frac{\hat{\sigma}_X}{\sqrt{n}}
    $$
    
    * 母集団平均の推定量$\bar{X}$には誤差があり，その正確性を示す。

---
**標本標準偏差**

* 確率変数の標本標準偏差：$\hat{\sigma}_X$
* 標本標準偏差の実現値：$\hat{\sigma}_x$

（注意）
* $\hat{\sigma}_X$は母集団標準偏差の不偏推定量では**ない**

---
**標本共分散**
* 確率変数の共分散

    $$
    \hat{\sigma}_{XY}=\frac{1}{n-1}\sum_{i=1}^{n}\left(X_i-\bar{X}\right)\left(Y_i-\bar{Y}\right)
    $$
    
* 共分散の実現値

    $$
    \hat{\sigma}_{xy}=\frac{1}{n-1}\sum_{i=1}^{n}\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)
    $$

（注意）
* 分母は $n-1$であり，これにより$\hat{\sigma}_X$は母集団共分散の不偏推定量となる。

---
**標本相関係数**
* 確率変数の相関係数

$$r_{XY}=\dfrac{\hat{\sigma}_{XY}}{\hat{\sigma}_X\cdot\hat{\sigma}_Y}$$

* 相関係数の実現値

$$r_{xy}=\dfrac{\hat{\sigma}_{xy}}{\hat{\sigma}_x\cdot\hat{\sigma}_y}$$

（注意）
* $r_{XY}$は母集団相関係数の不偏推定量では**ない**

