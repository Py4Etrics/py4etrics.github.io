#!/usr/bin/env python
# coding: utf-8

# # Pandas

# `Pandas`は`NumPy`の`array`のようにデータを扱うパッケージだが，Pandas特有のデータ構造を提供し複雑なデータを扱いやすくしている。例えば，行列にはラベルを使うことにより，インデックス番号だけではなくラベル名（例えば，GDP）を使って操作することが可能となる。Pandasには`DataFrame`（データフレーム）と`Series`（シリーズ）と呼ばれるオブジェクトがある。前者はエクセルのスプレッド・シートをイメージすれば良いだろう。後者は，スプレッド・シートから１つの行または列を取り出したようなデータと思えば良い。また，`Pandas`は`NumPy`に基づいているため，ベクトル演算（ブロードキャスティング）の機能が使える。
# 
# ここで説明できない他の使い方については[このサイト](https://github.com/ysdyt/pandas_tutorial)と[このサイト](https://note.nkmk.me/python-pandas-post-summary/)が参考になる。
# 
# 通常`pd`という名前で読み込む。

# In[1]:


import pandas as pd


# ## データの読み込みとデータのチェック

# ```{margin}
# <div name="html-admonition">
# Do you want to read in a differnt language? Open the 
# <input type="button" onclick="location.href='https://translate.google.com/translate?hl=&sl=ja&tl=en&u='+window.location;" value="Google translated version" style="color:#ffffff;background-color:#008080;" onmouseover="this.style.background='#99ccff'" onmouseout="this.style.background='#008080'"/>
# in English or the language of your choice.
# </div>
# ```

# 様々なデータを読み込むことが可能だが，ここでは`read_csv()`関数を使ってインターネット上の`.csv`ファイルを読み込む。

# In[2]:


# url の設定
url = 'https://raw.githubusercontent.com/Haruyama-KobeU/Py4Basics/master/data/data1.csv'

# 読み込み
df = pd.read_csv(url)


# `df`全体を表示させる。

# In[3]:


df


# 行ラベルがインデックス（番号）のままなので，列`year`を行ラベルに設定する。
# 
# * `set_index()`：選択された列を行ラベルにするメソッド

# In[4]:


df = df.set_index('year')
df


# ```{tip}
# * `df.set_index('year')`は直接`df`に影響を与えない。単に，書き換えるとどうなるかを表示している。ここでは`df`に再度割り当てることにより`df`自体を上書きしている。
# * 出力にある`NaN`（Not a Number）は欠損値を示す。
# * 行ラベルに`year`という列名が残るが，それを消す方法：
#     1. メソッド`.rename_axis('')`を使う。ここで`''`は空の文字列である。`.rename_axis(None)`でも同じ結果となる。
#     1. `df.index.name = ''`とする。ここで`''`は空の文字列
# ```

# 行数が大きい場合（例えば，10000），全てを表示してもあまり意味がない。そこでよく使うメソッドに最初や最後の数行だけを表示すものがある。
# 
# `df`の最初の５行を表示させる。

# In[5]:


df.head()


# 引数に2を指定すると最初の2行のみ表示される。

# In[6]:


df.head(2)


# 最後の5行を表示させる。引数に整数を入れて表示行数を指定することも可能。

# In[7]:


df.tail()


# `df`の情報を確認する。

# In[8]:


df.info()


# **説明**：
# * `<class 'pandas.core.frame.DataFrame'>`
#     * クラス名
#     * `type(1)`とすると`int`というデータ型が表示するが，これはクラス名でもある。`print(type(1))`とすると`<class 'int'>`と表示される。
# * `Int64Index: 11 entries, 2000 to 2010`
#     * 行のインデックスの情報
#     * データ型は`Int64`(整数）（データ型には`64`や`32`という数字がついている場合がある。それらは数字をコンピュータのメモリに記憶させる際，何ビット必要かを示している。より重要なのは`Int`（整数）の部分である。）
#     * 11個のデータで2000から2010
# * `Data columns (total 5 columns):`
#     * データ列の数（5つ）
# * `gdp  11 non-null int64`
#     * データ型は`int64`
#     * 11のデータがあり，欠損値なし（`non-null`とは欠損値ではないデータ）
# * `inv  10 non-null float64`
#     * データ型は`float64`
#     * 10のデータがあり，欠損値数は1（＝11-10）
# * `con  9 non-null float64`
#     * データ型は`float64`
#     * 9のデータがあり，欠損値数は2（＝11-9）
# * `pop  11 non-null int64`
#     * データ型は`int64`
#     * 11のデータがあり，欠損値数なし
# * `id   11 non-null object`
#     * データ型は`object`（文字列などの場合）
#     * 11のデータがあり，欠損値数なし
# * `dtypes: float64(2), int64(2), object(1)`
#     * `df`の列にどのようなのデータ型かを示す
#     * `float64`と`int64`が2列つずつ，文字列は１列
# * `memory usage: 528.0+ bytes`
#     * メモリー使用量は約528.0バイト
# 
# データを読み込んだら必ず`info()`を使って欠損値の数や列のデータ型を確認すること。
# 
# また，データの統計的な特徴は次のメソッドでチェックできる。

# In[9]:


df.describe()


# * `count`：観測値の数
# * `mean`：平均
# * `std`：標準偏差
# * `min`：最小値
# * `max`：最大値
# * `25%`：第１四分位数
# * `50%`：第２四分位数（中央値）
# * `75%`：第３四分位数
# * `max`：最大値
# 
# 
# 次のデータ属性を使って`df`の行と列の長さを確認することができる。返値はタプルで，`(行の数，列の数)`と解釈する。

# In[10]:


df.shape


# 返値はタプルなので，行数は以下で取得できる。

# In[11]:


df.shape[0]


# 以下でも行数を示すことができる。

# In[12]:


len(df)


# ## DataFrameの構成要素

# `DataFrame`には様々な属性があるが，ここでは以下の３点について説明する。
# 
# * データ（`df.to_numpy()`）
# * 列ラベル（`df.columns`）
# * 行ラベル（`df.index`）
# 
# まずデータ自体を抽出する。

# In[13]:


df.to_numpy()


# In[14]:


type(df.to_numpy())


# これで分かることは，メインのデータの部分は`NumPy`の`ndarray`（`n`次元`array`）であることが分かる。即ち，`Pandas`は`NumPy`に基づいて構築されており，データ値の計算などは`array`が裏で動いているということである。また行と列のラベルを追加し，より直感的に使えるように拡張しているのである。

# 次に列ラベルを取り出してみる。

# In[15]:


df.columns


# `dtype='object'`から列ラベルに使われているデータ型（`dtype`）はオブジェクト型（`object`）だとわかる。
# 
# * オブジェクト型とは文字型を含む「その他」のデータ型と理解すれば良いだろう。
# * `dtype='object'`と`dtype=object`は同じ意味。
# 
# 列ラベル自体のクラスは次のコードで調べることができる。

# In[16]:


type(df.columns)


# `dir()`もしくは`see()`で調べると多くのメソッドや属性が確認できるが，その中に`.tolist()`が含まれており，これを使うことにより列ラベルをリストに変換することができる。

# In[17]:


df_columns = df.columns.tolist()
df_columns


# 行ラベルについても同じことができる。

# In[18]:


df.index


# 行ラベルのデータ型`dtype`は整数である`int64`。列`year`を行ラベルに指定したため，`name='year'`はその列ラベルを表示している。行ラベルのデータ型（クラス）は

# In[19]:


type(df.index)


# であり，ラベルをリストとして抽出することもできる。

# In[20]:


df_index = df.index.tolist()
df_index


# ## 要素の抽出

# `NumPy`の`array`の場合，`[,]`を使い要素を抽出した。`Pandas`の場合，様々な抽出方法があるが，覚えやすく少しでも間違いの可能性を減らすために，そして可読性向上のために`array`に対応する以下の２つの方法を使うことにする。
# 
# * ラベルを使う方法：`.loc[,]`
# * インデックスを使う方法：`.iloc[,]`（これは`array`の`[ ]`と同じと考えて良い）
# 
# １つ目の`loc`はラベルのlocationと覚えよう。２つ目はの`iloc`の`i`はインデックス（index）の`i`であり，index locationという意味である。使い方は`array`の場合と基本的に同じである。
# 
# * `,`の左は行，右は列を表す。
# * 行または列を連続して選択する（slicing）場合は`:`を使う。（`start:end`）
#     * `:`の左右を省略する場合は，「全て」という意味になる。
#     * `:`の左を省略すると「最初から」という意味になる。
#     * `:`の右を省略すると「最後まで」という意味になる。
#     * `.loc[,]`の場合，`end`を含む。（要注意！）
#     * `.iloc[,]`の場合，`end`は含まず，その１つ前のインデックスまでが含まれる。
# * `,`の右に書く`:`は省略可能であるが省略しないことを推奨する。
# 
# 「特例」として`.loc[,]`と`.iloc[,]`以外に
# * ラベルと`[]`だけを使い**列**を選択する方法
# 
# も説明する。

# ```{warning}
# * `.loc[,]`の場合，`end`を含む。（要注意！）
# * `.iloc[,]`の場合，`end`は含まず，その１つ前のインデックスまでが含まれる。
# ```

# ### `.loc[,]`（ラベル使用）

# **１つの行を`Series`として抽出**

# In[21]:


df.loc[2005,:]


# **１つの行を`DataFrame`として抽出**

# In[22]:


df.loc[[2005],:]


# **複数行を抽出**

# In[23]:


df.loc[[2005, 2010],:]


# **複数行を連続抽出（slicing）**

# In[24]:


df.loc[2005:2008,:]


# **１つの列を`Series`として抽出**

# In[25]:


df.loc[:,'gdp']


# **複数列を抽出**

# In[26]:


df.loc[:,['gdp','pop']]


# **複数列を連続抽出（slicing）**

# In[27]:


df.loc[:,'inv':'pop']


# ### `.iloc[]`（インデックス使用）

# **１つの行を`Series`として抽出**

# In[28]:


df.iloc[1,:]


# **複数行を抽出**

# In[29]:


df.iloc[[1,4],:]


# **複数行を連続抽出（slicing）**

# In[30]:


df.iloc[1:4,:]


# **１つの列を`Series`として抽出**

# In[31]:


df.iloc[:,1]


# **１つの列を`DataFrame`として抽出**

# In[32]:


df.iloc[:,[1]]


# **複数列を選択**

# In[33]:


df.iloc[:,[1,3]]


# **複数列を連続抽出（slicing）**

# In[34]:


df.iloc[:,1:3]


# ### `[]`で列の選択（ラベル使用）

# **１つの列を`Series`として抽出**

# In[35]:


df['gdp']


# **１つの列を`DataFrame`として抽出**

# In[36]:


df[['gdp']]


# **複数列を選択**

# In[37]:


df[['gdp','pop']]


# ## ある条件の下で行の抽出

# ### １つの条件の場合

# #### 例１：GDPが100未満の行の抽出

# まず条件を作る。

# In[38]:


df['gdp'] < 100


# この条件では，GDPが100未満の行は`True`，以上の行は`False`となる。この条件を`cond`というの変数に割り当てる。`()`を省いても良いが，ある方が分かりやすいだろう。

# In[39]:


cond = ( df['gdp'] < 100 )


# `cond`を`.loc[,]`の引数とすることにより，`True`の行だけを抽出できる。（注意：`cond`を使って**行**を抽出しようとしているので`,`の左側に書く。）

# In[40]:


df.loc[cond,:]


# この条件の下で$inv$だけを抽出したい場合
# 
# * `df.loc[cond,'inv']`
# 
# とする。

# ```{warning}
# 以下のように抽出を連続ですることも可能だが，避けるように！
# * `df.loc[cond,:]['inv']`
# * `df.loc[cond,:].loc[:,'inv']`
# ```

# #### 例２：`id`が`a`の行を抽出

# In[41]:


cond = ( df.loc[:,'id'] == 'a' )
df.loc[cond,:]


# ### 複数条件の場合

# #### 例３

# 以下の条件の**両方**が満たされる場合：
# 
# * `gdp`が100以上
# * `inv`が30以下
# 
# それぞれの条件を作成する。

# In[42]:


cond1 = ( df['gdp'] >= 100 )
cond2 = ( df['inv'] <= 30 )


# ２つの条件が同時に満たされる条件を作成する。

# In[43]:


cond = ( cond1 & cond2 )


# `cond`を引数に使い行を抽出する。

# In[44]:


df.loc[cond, :]


# #### 例４

# 以下の条件の**どちらか**が満たされる場合：
# * `gdp`は200以上
# * `con`は60以下

# In[45]:


cond1 = ( df['gdp'] >= 200 )
cond2 = ( df['con'] <= 60 )
cond = ( cond1 | cond2 )

df.loc[cond, :]


# #### 例５

# 以下の条件の**どちらか**が満たされ
# * `gdp`は200以上
# * `con`は60以下
# 
# かつ以下の条件も**同時に**満たされる場合：
# * `id`が`a`と等しい

# In[46]:


cond1 = ( df['gdp'] >= 200 )
cond2 = ( df['con'] <= 60 )
cond3 = ( df['id'] == 'a' )
cond = ( (cond1 | cond2) & cond3 )

df.loc[cond, :]


# ### `query()`

# `query()`というメソッドでは文字列を使い行の抽出コードを書くことができる。これにより直感的なコード書くことが可能である。

# #### 例１の場合：

# In[47]:


df.query('gdp < 100')


# #### 例２の場合

# In[48]:


df.query('id == "a"')


# #### 例３の場合

# In[49]:


df.query('(gdp >= 100) & (inv <= 30)')


# #### 例４の場合

# In[50]:


df.query('(gdp >= 200) | (con <= 60)')


# #### 例５の場合

# In[51]:


df.query('(gdp >= 200 | con <= 60) & (id == "a")')


# ````{tip}
# 変数で条件を設定する場合`@`が必要になる。例えば，変数`z`という変数があるとしよう。
# 
# ```python
# z = 100
# ```
# 
# 変数`z`の値に基づいて行の抽出をする場合は次のようにする。
# 
# ```python
# df.query('gdp < @z')
# ```
# 
# {glue:}`glue0_txt`
# ````

# In[52]:


from myst_nb import glue
z = 100
glue0 = df.query('gdp < @z')
glue("glue0_txt", glue0)


# ## 列と行の追加と削除

# ### 列の追加 `[ ]`

# `[]`に列ラベルを使って列を抽出することができるが，`[]`は列の追加にも使えるので，ここではその使い方を説明する。まず，全ての行が`1.0`となる列を作成するとしよう。その場合，以下のようにする。

# In[53]:


df['Intercept'] = 1.0


# In[54]:


df.head(2)


# 次の例では既存の列から新たな列を作成する。まず１人当たりGDPの計算を計算し，それを変数`gdp_pc`に割り当てる。

# In[55]:


gdp_pc = df['gdp']/df['pop']
gdp_pc.head()


# これは`Series`であり，`GDPpc`として`df`に追加する。

# In[56]:


df['gdp_pc'] = gdp_pc


# In[57]:


df.head(2)


# ### 列の追加 `.loc[,]`

# `.loc[]`は行と列の抽出に使ったが，追加にも使える。`[]`と同じと考えれば良い。次の例では`pop`を2倍した列を追加している。

# In[58]:


df.loc[:,'2pop'] = 2*df['pop']


# ### 列の削除 `[ ]`

# In[59]:


del df['2pop']


# ### 列の削除 `drop()`

# `DataFrame`には`.drop()`というメソッドが用意されているので，それを使うことも可能である。
# 
# 下のコードの説明：
# * オプション`axis=`の値を`columns`の代わりに`１`でも可
# * コピーを作るだけなので，元のdfを書き換えたい場合は以下のどちらかが必要
#     * `df`に代入する
#     * オプション`inplace=True`（デフォルトは`False`）を追加する。

# In[60]:


df = df.drop(['Intercept','gdp_pc'], axis='columns')

# df.drop('Intercept', axis='columns', inplace=True)


# ### 行の追加 `.loc[,]`

# 行と列の抽出，そして列の追加に使ったが，行の追加にも使える。

# In[61]:


df.loc[2011,:] = [215, 100, 115, 22, 'b']


# In[62]:


df.tail(3)


# 上の例では，最初の4つの要素は整数として入力されたが，`df`の中では浮動小数点に変換されている。

# ### 行の削除 `drop()`

# これは列を削除する際に紹介したメソッド`.drop()`である。
# 
# * オプション`axis=`の値を`rows`の代わりに`0`でも可
# * コピーを作るだけなので，元のdfを書き換えたい場合は以下のどちらかが必要
#     * `df`に代入する
#     * オプション`inplace=True`（デフォルトは`False`）を追加する。

# In[63]:


df = df.drop(2011, axis='rows')

# df.drop(2011, axis=0, inplace=True)


# ## 欠損値の扱い

# `Pandas`では欠損値は`NaN`と表示されるが，`na`もしくは`null`と呼んだりもする。

# ### 欠損値の確認

# 欠損値があるかどうかの確認は，`df.info()`でもできるが，以下のメソッドを組み合わせることでも可能である。
# 
# * `isna()`：それぞれの要素について`NaN`の場合`True`を，そうでない場合は`False`を返す。（`DataFrame`の全ての要素が`True/False`となる。）
# * `sum(axis='rows')`：`df`の上から下に**行**（rows）を縦断して，それぞれの列の中にある`True`数える。
#     * `rows`は複数！（`0`でも可）
# * `sum(axis='columns')`：`df`の左から右に**列**（columns）を横断して，それぞれの行の中にある`True`を数える。
#     * `columns`は複数！（`1`でも可）
#     
# （注意）`sum()`の`axis`は「行を縦断」か「列を横断」かを指定する。

# In[64]:


df.isna().sum(axis='rows')


# `inv`と`con`に`NaN`があることがわかる。`axis='columns'`を設定すると`NaN`がある行を確認できる。

# In[65]:


df.isna().sum(axis='columns')


# ---
# `NaN`がある行だけを抽出したい場合がある。その場合はメソッド`any()`が役に立つ。
# 
# * `any(axis='rows')`：`df`の上から下に行（`rows`）を縦断して，それぞれの列の中で一つ以上`True`がある場合には`True`を，一つもない場合は`False`を返す。
#     * `rows`は複数！（0でも可）
# * `any(axis='columns')`：dfの左から右に列（`columns`）を横断して，それぞれの行の中で一つ以上`True`がある場合には`True`を，一つもない場合は`False`を返す。
#     * `columns`は複数！（1でも可）
# 
# （注意）`any()`の`axis`は「行を縦断」か「列を横断」かを指定する。

# In[66]:


cond = df.isna().any(axis='columns')
df.loc[cond,:]


# これで`NaN`がある行を抽出することができる。

# ### 欠損値がある行の削除

# 欠損値がある行を削除するには，`df`のメソッド`.dropna()`を使う。次のコードでは，欠損値がある**全て**の行を削除する。

# In[67]:


df.dropna()


# このメソッドは，欠損値を削除するとどうなるかを示すだけであり`df`自体は影響は受けない。`df`自体から`NaN`がある行を削除する場合は`inplace=True`のオプション（デフォルトでは`False`になっている）を加えて
# ```
# df.dropna(inplace=True)
# ```
# とするか，削除後の`df`を変数`df`に再度割り当てる。
# ```
# df = df.dropna()
# ```

# 一方で，ある列に`NaN`がある場合のみ行を削除したい場合もあるだろう。その場合は引数`subset`を使う。

# In[68]:


df.dropna(subset=['inv'])


# 列`con`に`NaN`が残っている。
# 
# ＜注意＞<br>
# `subset=['inv']`のように，`NaN`があるかを確認する列が１つであってもリスト`[]`で指定する。複数であれば，次のように列ラベルをリストとして並べれば良い。
# ```
# subset=['inv','id']
# ```
# 非常に便利な引数なので覚えておこう！

# ## 並び替え

# `df`の行を並び替えるにはメソッド`.sort_values()`を使う。引数には，並び替える基準となる列ラベルを指定し，デフォルトでは昇順となる。次のコードでは`gdp`に従って昇順に並び替えている。

# In[69]:


df.sort_values('gdp').head()


# 引数`ascending=False`を使うと降順になる。

# In[70]:


df.sort_values('gdp', ascending=False).head()


# 複数の列を基準として，昇順と降順を別々に指定することも可能である。

# In[71]:


df.sort_values(['id','gdp'], ascending=['True','False']).head()


# ここでは`id`に従って先に並び替えられ，その後に`gdp`に従って並び替えられている。`ascending`は昇順（`True`）か降順（`False`）かを指定する引数であり，`['id','gdp']`と`ascending=['True','False']`の順番が対応している。即ち，`id`は昇順，`gdp`は降順となるように並び替えている。

# ## DataFrameの結合

# データを扱うと必ずと言って良いほど必要になるのがデータ・セットの結合である。例えば，日本銀行と米国連邦準備制度（米国の中央銀行）のデータからなる`df`を作るとすると，それぞれのサイトから取得したデータ・セットを結合する必要がある。`DataFrame`には，そのための便利なメソッドが準備されているので，ここではそれを紹介する。
# 
# ３つの`.csv`ファイルを使う。

# In[72]:


url = 'https://raw.githubusercontent.com/Haruyama-KobeU/Py4Basics/master/data/'
df1 = pd.read_csv(url+'data1.csv')
df2 = pd.read_csv(url+'data2.csv')
df3 = pd.read_csv(url+'data3.csv')


# In[73]:


df1


# In[74]:


df2


# In[75]:


df3


# In[76]:


df3.loc[0,'year']


# ### 横結合：`merge()`

# `merge()`以外にも結合に使える関数はあるが，ここでは`merge()`のみを考える。
# 
# `df1`を「左」，`df2`を「右」に横結合する。
# ```
# pd.merge(df1, df2, on=None, how='inner')
# ```
# * `on`はどの列を基準にして結合するかを指定（ここでは「基準列」と呼ぼう）
#     * 例えば，`df1`と`df2`の両方に`year`の列がある場合，`on='year'`とすると列`year`が基準列となる。
#         * `df1`と`df2`の別々の列を指定する場合は`left_on=`と`right_on=`を使う。
#     * 基準列に基づいて残す行を決める（`how`で説明する）
#     * 基準列にある要素の順番が合ってなくても，自動でマッチさせる。
#     * 複数指定も可
#     * デフォルトは`None`
# * `how`は`on`で指定した基準列に基づいてどのように結合するかを指定
#     * `inner`：`df1`と`df2`の両方に基準列ある行だけを残す（デフォルト）。
#     * `left`：`df1`の行は全て残し，`df2`からはマッチする行だけが残る（マッチしない行は削除）。また`df1`にあり`df2`にない行には`NaN`が入る。
#     * `right`：`df2`の行は全て残し，`df1`からはマッチする行だけが残る（マッチしない行は削除）。また`df2`にあり`df1`にない行には`NaN`が入る。
#     * `outer`：`df1`と`df2`の両方の行を残し，マッチする行がない場合は`NaN`を入れる。
# 
# 
# （コメント）
# この他に様々な引数があるので[このサイト](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html)を参照。例えば，場合によっては次の引数を使う必要があるかも知れないので確認しよう。
# * `left_index=True`：`df1`では行インデックスを基準列にする。
# * `right_index=True`：`df2`では行インデックスを基準列にする。

# #### 例１

# * `year`を基準列とする。
# * 共通する行だけを残す。

# In[77]:


pd.merge(df1, df2, on='year', how='inner')


# #### 例２

# * `year`を基準列とする。
# * `df1`の全ての行を残す。
# * `df2`ではマッチする行だけ残り，マッチしない行は削除される。
# * `df1`にあり`df2`にない行には`NaN`が入る。

# In[78]:


pd.merge(df1, df2, on='year', how='left')


# #### 例３

# * `year`を基準列とする。
# * `df2`の全ての行を残す。
# * `df1`ではマッチする行だけ残り，マッチしない行は削除される。
# * `df1`にあり`df2`にない行には`NaN`が入る。

# In[79]:


pd.merge(df1, df2, on='year', how='right')


# #### 例４

# * `year`を基準列とする。
# * `df1`と`df2`の全ての行は残る。
# * `df1`にあり`df2`にない行には`NaN`が入る。
# * `df2`にあり`df1`にない行には`NaN`が入る。

# In[80]:


pd.merge(df1, df2, on='year', how='outer')


# ### 縦結合

# `concat()`は横結合にも使えるが，縦結合のみ考える。
# 
# 引数には複数の`DataFrame`をリストとして書く。

# In[81]:


df13 = pd.concat([df1,df3])


# In[82]:


df13.tail()


# ## その他

# ### インデックスを振り直す

# メソッド`.reset_index()`を使うと，行のインデックスを0,1,2,..と振り直すことができる。`df13`を使い説明する。

# In[83]:


df13.reset_index()


# この例では元のインデックスが列`index`として追加されている。`reset_index()`に引数`drop=True`を加えると，列`index`は追加されない。

# In[84]:


df13.reset_index(drop=True).head()


# ### 列のラベルの変更

# メソッド`.rename()`を使い列のラベルを変更する。引数は次の形で設定する。
# 
# $$\text{.rename}\left(\text{columns=}辞書\right)$$
# 
# ここで「辞書」は次のルールで指定する。
# * `key`:元のラベル
# * `value`：新しいラベル
# 
# 下のコードでは，`df13`を使い新しいラベルとして`pop_new`と`id_new`を使っている。

# In[85]:


df13.rename(columns={'pop':'pop_new','id':'id_new'}).head()


# `.rename()`以外にリストを使って列ラベルを変更することも可能である。例えば，
# ```
# df.columns = ['year', 'gdp', 'inv', 'con', 'pop_new', 'id_new']
# ```
# この場合，列の数とリストの要素数が合わないとエラーとなるので注意しよう。

# ### 列の並び替え

# #### アルファベット順

# メソッド`.sort_index()`を使い，引数には`axis='columns'`もしくは`axis=1`を指定する。
# 
# （コメント）引数が`axis='rows'`もしくは`axis=0`の場合，行が並び替えられる。

# In[86]:


df13.sort_index(axis='columns').head()


# #### 順番を指定する

# ##### 方法１

# 列を選択する方法を使い並び替える。

# In[87]:


var = ['id','year','gdp','con','inv','pop']

df13.loc[:,var].head()


# ##### 方法２

# 次の列の選択方法でも可。

# In[88]:


df13[var].head()


# ##### 方法３

# メソッド`.reindex()`を使う。引数は`columns=[]`。

# In[89]:


df13.reindex(columns=['id','year','gdp','con','inv','pop']).head()


# ##### 方法３の応用

# 最後の行を最初に移動する。
# * `.columns.tolist()`：コラムのラベルを取得し，それをリストに変換
# * `[col[-1]]+col[0:-2]`の分解と結合
#     * `[col[-1]]`：`col`の最後の要素を抽出するが，文字列として返されるので（外側の）`[ ]`を使ってリストに変換
#     * `col[0:-2]`：`col`の最初から最後から二番目の要素をリストとして取得
#     * `+`：リストの結合

# In[90]:


col = df13.columns.tolist()

new_col = [col[-1]]+col[0:-2]

df.reindex(columns=new_col).head()

