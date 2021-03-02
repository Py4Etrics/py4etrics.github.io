---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 計量経済学用語

If you come here without expecting Japanese, please click [Google translated version](https://translate.google.com/translate?hl=&sl=ja&tl=en&u=https%3A%2F%2Fpy4etrics.github.io%2F7_Review_of_Statistics.html) in English or the language of your choice.

---

* 横断面データ（cross-sectional data）
    * ある時点において複数の対象を観察し収集したデータ（例えば，2018年の47都道府県の県内総生産）
* 時系列データ（time-series data）
    * ある項目について時間に沿って集めたデータ（例えば，1950~2018年の兵庫県の県内総生産の年次データ）
* パネル・データ（panel ata）
    * 横断面データと時系列データの両方の特徴を兼ね備えたデータ。（例えば，1990~2018年の47都道府県の県内総生産データ）
* 観測値 ＝ observations (observed values)
    * データの値（例えば，価格や人数）
    * a value of individual data
* 標本（サンプル）＝ sample 
    * 観測値で構成されるデータまたはデータ・セット
    * a set of data
* 標本の大きさ ＝ sample size ＝ サンプル・サイズ
    * 母集団から$n$個の観測値を収集したとすると，$n$＝「標本の大きさ」
* 標本数 ＝ number of samples
    * 母集団から$n$個の観測値の標本を$N$組収集したとすると，$N$＝「標本数」
    * $N$の標本数がある場合，個々の標本の大きさを$n_i$, $i=1,2,3,...N$と書くこともできる。
* 統計量 ＝ statistics
    * ２つの意味で使われる
        1. 単純に標本データの特徴を要約するために使われる**関数**。平均の式が一例であり、データが「入力」であり、その平均が「出力」。標本によって異なるため確率変数（random variables）
        1. 上の関数を使い計算した数値
* 推定 ＝ estimation
    * 母集団の未知のパラメータを計算し推知すること
* 推定量 ＝ estimators
    * **母集団のパラメータ**（母数）を推定するために用いる関数としての統計量（例えば，回帰直線の係数の式）
        * 母集団のパラメータという真の値と比較してバイアスを考えることができるのが推定量であるが、統計量はそうではない
    * 標本によって異なるため確率変数（random variables）
* 推定値 ＝ estimates
    * 母集団のパラメータを推知するために推定量を使い実際に計算して得た数値（例えば，回帰の定数項）
    * 確率変数の実現値
* 母集団回帰式 ＝ Population Regression Equation (PRE)
    * 母集団の値を生成する回帰式（観測不可能）
    
        $$y = \beta_0 + \beta_1 x + u$$
    
* データ生成過程（Data Generating Process; DGP）
    * データが生成される仕組み
    * 母集団回帰式と同義
* 誤差項（攪乱項）＝ (random) errors
    * 母集団回帰式にある観測できないランダム変数（PREの$u$）
* モデル = models
    * 母集団のパラメータを推定するために使う回帰式
    * 最小二乗法推定法は推定方法でありモデルではないので注意しよう（「OLSモデル」は存在しない）。
* 残差 ＝ residuals
    * 回帰式の被説明変数の観測値と予測値の差（次式の$\hat{u}$）
    
        $$ \hat{u}_i=y_i-\hat{\beta}_0-\hat{\beta}_1x_i$$
    
    * $\hat{\beta}_0$, $\hat{\beta}_0$はデータを使って計算した推定量
    * $i=1,2,...n$（$n$は標本の大きさ）
