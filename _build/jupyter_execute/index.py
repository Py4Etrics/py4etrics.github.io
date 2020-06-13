# Pythonで学ぶ入門計量経済学

```{epigraph}
[春山 鉄源](https://t-haruyama.github.io)

神戸大学経済学研究科
```

import datetime
dt = datetime.datetime.now()
print('Version:',dt.strftime('%Y年%m月%d日 %H:%M:%S'))

本書に関するコメントなどは<haruyama@econ.kobe-u.ac.jp>までご連絡ください。

## はじめに

「なぜプログラミング？」文系の経済学の学生が理系のプログラミングを学ぶとなると，まず頭に浮かぶ質問かも知れない。過去にも同じような質問を問うた経済学部の卒業生は多くいると思われる。例えば，Excelのようなスプレッドシートのソフトは1980年代からあり，当時の大学生も使い方を学ぶ際「なぜ？」と思ったことだろう。しかし今ではWord，Excel，PowerPointの使い方は，大学卒業生にとって当たり前のスキルになっている。同じように，AI（人工知能）やビッグデータが注目を浴びる社会では，ある程度のプログラミング能力も経済学部卒業生にとって当たり前のスキルになると予測される。実際，文系出身でプログラミングとは縁のなかったある大手新聞社の記者の話では，社内で`Python`の研修を受けており，仕事に活かすことが期待されているという。また，プログラミングを学ぶ重要性を象徴するかのように，2020年度からは小学校でプログラミング的思考を育成する学習指導要領が実施され，続いて中高教育でもプログラミングに関する内容・科目が充実される予定である。このようにプログラミングのスキルの重要性は益々大きくなると思われる。

「なぜ`Python`？」プログラミング言語は無数に存在し，それぞれ様々な特徴があり，お互いに影響し合い進化している。その過程で，広く使われ出す言語もあれば廃れていく言語もある。その中で`Python`は，近年注目を集める言語となっている。それを示す相対的な人気指標として[Stack Overflow Trends](https://insights.stackoverflow.com（/trendstに追いつくには年数を重ねる必要があるだろうags=java%2Cc%2B%2B%2Cpython%2Cc%23%2Cjavascript%2Cassembly%2Cphp%2Cr%2Cruby%2Cruby-on-rails&fbclid=IwAR3nGm3GtZT99DnP6Ov-hW0-TmZoeePajsLOyRxZYSUjV3F8qbeiOcG-oak)がある。
```{figure} /image/many_languages.png
:scale: 35%
```
Stack Overflowとは，プログラミングに関する質問をすると参加者が回答するフォーラムであり，質の高い回答で定評があるサイトである。その英語版で，ある言語に関する質問が何％を占めるのかを示しているのが図１である。2012年頃から`Python`は急上昇しているが（右上がりの赤色の線），過去５年間でみると下降トレンドの言語が多い印象である。

では，`Python`の人気はどこにあるのか？まず最初の理由は無料ということである。経済学研究でよく使われる数十万円するソフトと比べると，その人気の理由は理解できる。しかし計量経済学で広く使われる`R`を含めて他の多くの言語も無料であり，それだけが理由ではない。人気の第２の理由は，汎用性である。`Python`はデータ分析や科学的数値計算だけではなく，ゲーム（ゲーム理論ではない），画像処理や顔認識にも使われている。またPCやスマートフォンの様々な作業の自動化に使うことも可能なのである。第３の理由は，学習コストが低いことである。`Python`のコードは英語を読む・書く感覚と近いため，他の言語と比較して可読性の高さが大きな特徴である。もちろん，`Python`の文法や基本的な関数を覚える必要があるが，相対的に最も初心者に易しい言語と言われる程である。他にも理由はあるが，`Python`はIT産業だけではなく金融・コンサルティング・保険・医療などの幅広い分野で使われており，データ分析の重要性が増すごとにより多くの産業で使われると思われる。

`Python`は経済学でどれ程使われているのだろうか。残念ながらデータはないが，私の個人的な印象では，数値計算に優れている`Matlab`（有料）や`Scilab`（無料），計量経済学では`R`（無料）や`Stata`（有料）が広く使われているようであり，その中で`Python`は比較的にマイナーであった。しかし近年それが変わる気配がある。1995年にノーベル経済学賞を受賞したRobert Lucas,JrとJohn Stachurskiが始めた[QuantEcon](https://quantecon.org/)では，`Python`と今後有望視される`Julia`のパッケージが公開され，高額な有料ソフトである`Matlab`でできることが可能になっている。また講義ノートや様々なコードも公開され，特に若い研究者は大きな影響を受けるのではないかと予想される。更には，経済学関連分野のファイナンスやデータ・サイエンス（例えば，機械学習）の分野では`Python`は広く使われている。

経済学部生が`Python`を学ぶ意義はどこにあるのだろうか。図１のデータはディベロッパーや様々な学問・業種の人々の行動を反映しており，経済学に携わる人の中での人気を表している訳ではない。しかし経済学部の大多数の卒業生は幅広い産業で働くことになる。社会全体で注目され，今後より多くの産業で使われることが予測される言語を学ぶことは有意義ではないだろうか。

このような背景の下，本書の目的は基本的な`Python`の使い方を学び，入門レベルの計量経済学を`Python`を使って学ぶことである。換言すると，計量経済学を通して`Python`の使い方を学ぶことである。本書を通して学ぶ`Python`の知識は，他分野への応用に大いに役立つと期待される。また`Python`を使い推定だけではなくシミュレーションもおこなうため，計量経済学の教科書で学ぶ推定量の性質などを直に感じ取り，計量経済学の復習にも役立つであろう。第１部では，`Python`の基礎を学び，第２部では計量経済分析に`Python`をどのように使うかを解説する。第３部では番外編として，関連するトピックに言及する。

## 参考書
本書では世界的に有名な学部レベルの教科書であるWooldridge (2019)を参考にしている。またこの本で扱われるデータを使い推定をおこなう。

> Introductory Econometrics: A Modern Approach, 7th ed, by J.M. Wooldridge

残念ながら邦訳はないが，読者には是非英語で読んで欲しい教科書である。

## 本書で使うPythonとパッケージのバージョン

import linearmodels, lmdiag, matplotlib, numba, numpy, pandas, py4etrics, scipy, see, statsmodels, wooldridge
from platform import python_version

packages = ['Python','linearmodels', 'lmdiag','matplotlib', 'numba', 'numpy','pandas', 'py4etrics', 'scipy','see', 'statsmodels', 'wooldridge']
versions = [python_version(),linearmodels.__version__, '0.3.7', matplotlib.__version__, numba.__version__, numpy.__version__, pandas.__version__, py4etrics.__version__, scipy.__version__, see.__version__, statsmodels.__version__, wooldridge.__version__]

for pack, ver in zip(packages, versions):
    print('{0:14}{1}'.format(pack,ver))

---

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

xx = np.linspace(-2.75,2.75,100)
plt.xkcd()
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2,sharey=ax1)
ax1.plot(xx, norm.pdf(xx,scale=1), 'k')
ax1.set_title('Normal Distribution', size=25)

ax2.plot(xx, norm.pdf(xx,scale=1), 'k')
ax2.scatter(-0.4, 0.28, s=300, linewidth=2.5, facecolors='none', edgecolors='k')
ax2.scatter(0.4, 0.28, s=300, linewidth=2.5, facecolors='none', edgecolors='k')
ax2.plot(xx,-0.02*np.cos(3*xx), 'k')
ax2.set_title('Paranormal Distribution', size=25)
plt.show()

from myst_nb import glue
glue("paranormal", fig, display=False)

````{toggle} Normal vs Paranormal
```{glue:} paranormal
```
````


````{toggle}
```
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

xx = np.linspace(-2.75,2.75,100)
plt.xkcd()
fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2,sharey=ax1)

ax1.plot(xx, norm.pdf(xx,scale=1), 'k')
ax1.set_title('Normal Distribution', size=25)

ax2.plot(xx, norm.pdf(xx,scale=1), 'k')
ax2.scatter(-0.4, 0.28, s=300, linewidth=2.5, facecolors='none', edgecolors='k')
ax2.scatter(0.4, 0.28, s=300, linewidth=2.5, facecolors='none', edgecolors='k')
ax2.plot(xx,-0.02*np.cos(3*xx), 'k')
ax2.set_title('Paranormal Distribution', size=25)
plt.show()
```
````



```{toctree}
:hidden:
:titlesonly:


part_1
part_2
part_3
```