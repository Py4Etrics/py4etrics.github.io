# Matplotlib

Matplotlib（マットプロットリブ）は図を描くためのパッケージである。どこまで手の込んだ図を書くかによってコードが違ってくるが，ここでは`pyplot`というサブパッケージを使い，一番シンプルなコードになるものを紹介する。ここで説明することができない多くの機能が備わっているが，この[リンク](https://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)が参考になるだろう。

通常，`plt`として読み込む。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 図の作成

`df`を再度読み直す。今回は`year`を行ラベルにせず，インデックス番号をそのまま使う。

url = 'https://raw.githubusercontent.com/Haruyama-KobeU/Haruyama-KobeU.github.io/master/data/data1.csv'
df = pd.read_csv(url).dropna()
df.head(3)

`gdp`を縦軸にした図を描く。**Jupyter Notebook**を使うと最小限必要なコードは１行となる。

plt.plot('gdp', data=df);

* `()`内の最初の変数は縦軸の変数，次はデータの指定
* デフォルトは線グラフ
* 行の最後に`;`があることに注意。これは「以降の出力をストップ」といいう意味である。
    * `;`は省略可能。省略すると特に必要でないものが示される（試そう！）

`;`を使う代わりに
```
plt.plot('gdp', data=df)
pass
```
もしくは
```
_ = plt.plot('gdp', data=df)
```
としても同じ結果となる。`pass`は文字通り「パスする」のパスであり。`_`を単独で使う場合，必要でない値を代入する変数によく使われる記号である。

上の図の不満な点は，横軸が不明なこと（単に`gdp`のデータ数に従って1,2,..と番号が振られている）。横軸の変数を指定するには，`()`内に追加するだけである。ただ順番に気をつけること。

plt.plot('year', 'gdp', data=df);

コードをみて分かるように
* `()`内の最初の変数は横軸の変数，２番目の変数は縦軸の変数，３番目はデータの指定

３番目のデータ指定は「このデータの中にある`year`と`gdp`を使う」ということを意味しており，データ指定がなければ`Python`はどの`year`と`gdp`か分からずにエラーとなる。一方で，データ指定をせずに，直接横軸・縦軸の変数を指定することも可能である。以下がその例：

plt.plot(df['year'], df['gdp']);

上の図で何をプロットしているかを分かっている場合はこれで十分だが，論文などに使う場合は不十分である。以下では「飾り付け」をする。

plt.plot('year', 'gdp',        # 横軸の変数，　　縦軸の変数
         color='red',         # 色 ：　赤
         linestyle='dashed',  # 線のタイプ：点線
         marker='o',          # マーカー：点
         data=df)             # データの指定
plt.xlabel('year')            # 横軸のラベル
plt.ylabel('GDP')             # 縦軸のラベル
plt.title('Gross Domestic Product')  # 図のタイトル
plt.grid()                    # グリッドの表示

様々なオプションが用意されている


|色  | 省略形|
|:---|:---:|
|blue | b  |
|green | g |
|red | r   |
|cyan | c  |
|magenta | m |
|yellow | y |
|black | k |
|white | w |


|線のスタイル | 説明 |
|:---:|:---------|
|-  | solid line style |
|-- |dashed line style |
|-. | dash-dot line style |
|:  | dotted line style |

|マーカー | 説明 |
|:------:|:----|
|.	| point marker |
|,	| pixel marker |
|o	| circle marker |
|v	| triangle_down marker |
|\* | star marker |
|+	|plus marker |
|x	| x marker |

数多くのオプションがある。[ここを参照](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)。

このようなオプションも簡略化して書くこともできる。例えば，上のコードで
```
color='red',
linestyle='dashed',
marker='o',
```
の３行を以下の一行にまとめることも可能である。
```
'r--o',
```

**複数の図**

単に`plt.plot()`を付け加えるだけ。

plt.plot('year', 'gdp', data=df)
plt.plot('year', 'con', data=df)
plt.plot('year', 'inv', data=df)
plt.legend();   # 凡例

**散布図の描画**

plt.scatter('gdp', 'con', data=df)
plt.xlabel('GDP')  #　横軸のラベル （省略可）
plt.ylabel('Consumption');  # 縦軸のラベル（省略可）

**ヒストグラム**

plt.hist(df['gdp'])
plt.title('GDP')   # 省略可
pass

**パイチャート**

lab = ['Investment', 'Consumption']  # ラベルの作成
dt = df.loc[1,['inv','con']]  # 2001年のデータを取り出す
plt.pie(dt,    # データの指定
        labels=lab,  # ラベルの指定 （省略可だが，ある方がわかりやすい）
        autopct='%.2f')   # ％表示  （省略可）
plt.title('GDP Share in 2001')   # タイトル （省略可）
pass

**ボックスプロット**

解釈については[このサイト](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)を参照。

plt.boxplot(df['gdp'])
plt.title('GDP')   # 省略可
pass

## 複数の図を並べて表示

複数の図を「１つの図」として描画するために`subplot(a,b,c)`を使い，行と列を指定してそれぞれの図の位置を設定する。
* `a`：行の数
* `b`：列の数
* `c`：行・列を使って位置を指定

＜例：２×２の場合＞

`subplot(2,2,1)`: 左上の位置を指定

`subplot(2,2,2)`: 右上の位置を指定

`subplot(2,2,3)`: 左下の位置を指定

`subplot(2,2,4)`: 右下の位置を指定

x = np.linspace(-2,2,100)

plt.figure(figsize=(10, 8))  # 図の大きさを設定（省略可）

# n = 5
plt.subplot(221)
plt.plot(x, x)
plt.title('A Positive Slope')

# n = 10
plt.subplot(222)
plt.plot(x, -x)
plt.title('A Negative Slope')

# n = 100
plt.subplot(223)
plt.plot(x, x**2)
plt.title('A U Shape')

# n = 1000
plt.subplot(224)
plt.plot(x, -x**2)
plt.title('An Inverted U Shape')
pass

## 図の保存方法

例として，図を`png`ファイルで保存する場合を考えよう。

＜開いている`Jupyter Notebook`のファイルと同じフォルダーに保存する場合＞

```
plt.savefig('<ファイル名.png')
```

を使う。この場合`;`や`pass`を使を必要はない。

plt.plot('gdp', data=df)
plt.savefig('gdp.png')

保存できる画像ファイルの種類には以下がある。

`png`，`jpg`，`pdf`，`ps`，`eps`，`svg`

これらのフォーマットで保存する場合は，上のコードの`png`を`pdf`等に入れ替える。

---
＜Mac: 開いている`Jupyter Notebook`のファイルのサブフォルダーフォルダーに保存する場合＞

例えば，サブフォルダー`temp`に画像を保存したい場合は
```
plt.savefig('./temp/<ファイル名.png')
```
とする。ここで`.`が開いている`Jupyter Notebook`のファイルがあるフォルダーを表している。