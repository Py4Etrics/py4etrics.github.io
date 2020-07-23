# 成績分析

## データの読み込み

import pandas as pd
import matplotlib.pyplot as plt
from see import see

### 学生

「うりぼー」→ 成績修得情報　→　成績表を選択コピー

```
df = pd.read_clipboard()
df.head()
```

# csvに保存

# df.to_csv('transcript20200708.csv')

# 保存先のフォルダの確認

# %pwd

### 教員

df = pd.read_csv('https://raw.githubusercontent.com/Haruyama-KobeU/Py4Basics/master/data/data_for_mark.csv')
df

## 列の内容の確認

ステップ１：`for`ループによる列のラベルの表示

col = df.columns
col

for c in col:
    print(c)

ステップ２：それぞれの列の要素の種類を表示

df['区分'].unique()

スッテプ３：スッテプ１と２を同時に

for c in col[1:]:
    x = df[c].unique()
    print(c, x)

### `DataFrame`の作成

df_other = df.query('区分 == "全学共通授業科目" or 区分 == "高度教養科目"')
df_econ = df.query('区分 == "専門科目"')

len(df),len(df_other),len(df_econ)

全角文字を使わない方法

kubun = df.loc[:,col[1]]

kubun_arr = kubun.unique()
kubun_arr

other = ( ( kubun == kubun_arr[0] ) | ( kubun == kubun_arr[1] ) )
econ = ( kubun == kubun_arr[2] )

df_other = df.loc[other,:]
df_econ = df.loc[econ,:]

len(df),len(df_other),len(df_econ)

## 全科目

### `f-string`

x='春山'
print(f'私は{x}ゼミに所属しています。')

l = [1,2,3]
print(f'合計は{sum(l)}です。')

### 優・秀・良・可・不可などの数

#### 簡単な方法

`value_counts()`を使うと簡単になる。

df.loc[:,'評語'].value_counts()

#### 表示を整理したい場合

評語の種類

m = df.loc[:,'評語'].unique()
m

mark = [m[2],m[3],m[4],m[1],m[0],m[-1],m[-2],m[-3]]
mark

for m in mark:
    print(m)

print('--------\n合計')

lst = []

for m in mark:
    no = len(df.query('評語 == @m'))
    lst.append(no)

lst

lst = []

for m in mark:
    no = len(df.query('評語 == @m'))
    lst.append(no)
    print(m, no)

print('--------\n合計',sum(lst))

### 優・秀・良・可・不可などの％

#### 簡単な方法

(100 * df.loc[:,'評語'].value_counts() / len(df.loc[:,'評語']) ).round(1)

#### 表示を整理したい場合

lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)

lst

lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, f': {percent}')
    
print(f'-----------\n合計: {sum(lst)}')

lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, f': {percent:.1f}')
    
print(f'-----------\n合計: {sum(lst):.0f}')

上のコードでは`f-string`を使った。その代わりに`format()`を使うことも可能。

lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, '{:.1f}'.format(percent))
    
print('----------\n合計 {:.0f}'.format(sum(lst)))

## 全学共通授業科目

### 数

df_other.loc[:,'評語'].value_counts()

lst = []

for m in mark:
    no = len(df_other.query('評語 == @m'))
    lst.append(no)
    print(m,no)
    
print(f'--------\n合計 {sum(lst)}')

### 割合

(100 * df_other.loc[:,'評語'].value_counts() / len(df_other.loc[:,'評語']) ).round(1)

lst = []

for m in mark:
    percent = 100 * len(df_other.query('評語 == @m')) / len(df_other)
    lst.append(percent)
    print(m, f'{percent:.1f}')
    
print(f'-----------\n合計 {sum(lst):.0f}')

## 専門科目

df_econ.loc[:,'評語'].value_counts()

lst = []

for m in mark:
    no = len(df_econ.query('評語 == @m'))
    lst.append(no)
    print(m,no)
    
print(f'--------\n合計 {sum(lst)}')

(100 * df_econ.loc[:,'評語'].value_counts() / len(df_econ.loc[:,'評語']) ).round(1)

lst = []

for m in mark:
    percent = 100 * len(df_econ.query('評語 == @m')) / len(df_econ)
    lst.append(percent)
    print(m, f'{percent:.1f}')
    
print(f'--------\n合計 {sum(lst):.0f}')

## GPAの推移

### 「科目GP」で記号がある行の削除

科目GPの要素の種類

df.loc[:,'科目GP'].unique()

全て文字列となっているので，`-`と`*`の記号が含まれない行だけから構成される`DataFrame`を作成する。

後で「科目GP」のデータ型を変更する際に警告がでないようにメソッド`.copy()`を使い`DataFrame`のコピーを作成する。

gpa = df.query("科目GP not in ['-', '*']").copy()
gpa.head()

gpa['科目GP'].unique()

データ型は`object`（文字列）のままである。

属性`.dtypes`を使って確認することもできる。

gpa['科目GP'].dtypes

`O`は`object`。

### 「科目GP」を浮動小数点に変更

科目GPは`object`（文字列）となっているので，メソッド`astype()`を使って`float`に変換する。

gpa['科目GP'] = gpa['科目GP'].astype(float)

gpa.dtypes

### 図示：毎年

#### `groupby`

`groupby`は`DataFrame`や`Series`をグループ化し，グループ内の計算を簡単に行うことができる便利なメソッドである。「修得年度」でグループ化し，平均を計算する。

gpa_grouped = gpa.groupby('修得年度')
gpa_mean = gpa_grouped.mean()
gpa_mean

* `gpa_grouped`は`DetaFrame`ではないので注意しよう。
* `gpa_grouped`には`median()`，`std()`，`var()`，`min()`，`max()`などが使える。その他の属性・メソッドを確認。

see(gpa_grouped)

#### 図示

方法１：横軸に文字列を使う

yr = [f'{i}' for i in gpa_mean.index]
yr

plt.plot(yr,gpa_mean['科目GP'], 'o-')
pass

方法２：`plt.xticks()`を使って

横軸を調整するために，`gpa_mean`のインデックスを確認。

yr = gpa_mean.index
yr

`xticks()`を使って横軸の表示を指定する。

plt.plot('科目GP', 'o-', data=gpa_mean)
plt.xticks(yr)
pass

### 図示：前期後期毎

#### 「修得学期」の英語化

「修得学期」の要素が日本語だと警告が出るため，メソッド`.replace()`を使って英語に変換する。`replace()`の引数は辞書で指定する。
* キー：変更する対象の数値や文字列
* 値：変更後の数値や文字列

gpa['修得学期'] = gpa['修得学期'].replace({'前期':'1','後期':'2'})
gpa.head()

#### `groupby`

「修得年度」と「修得学期」でグループ化する。

gpa_grouped = gpa.groupby(['修得年度','修得学期'])
gpa_mean = gpa_grouped.mean()
gpa_mean

図示する際エラーをがでないようにインデックスのラベルを削除する。

gpa_mean.index.names = (None,None)
gpa_mean

#### 図示

横軸に文字列を使う。

gpa_mean.index

yr_half = [f'{i[0]}-{i[1]}' for i in gpa_mean.index]
yr_half

plt.plot(yr_half, gpa_mean['科目GP'])
pass