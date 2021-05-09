#!/usr/bin/env python
# coding: utf-8

# # 成績分析

# If you come here without expecting Japanese, please click [Google translated version](https://translate.google.com/translate?hl=&sl=ja&tl=en&u=https%3A%2F%2Fpy4etrics.github.io%2FGrade.html) in English or the language of your choice.
# 
# ---

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from see import see


# 授業の成績（優秀良可不可）とGP（Grade Point）を使った成績の分析を行う。成績表のコピーは神戸大学の「うりぼーネット」を用いる例を考えるが、Web上で成績表が表示されるのであればコードを少し修正するだけで分析が行えるであろう。

# ## データの読み込み

# ### 学生

# 「うりぼー」→ 成績修得情報　→　成績表を選択しコピー

# In[2]:


#ーーーーー dfに成績表を割り当てる ーーーーー

# df = pd.read_clipboard()
# df.head()


# In[3]:


#ーーーーー csvに保存 ーーーーー

# df.to_csv('成績表20200708.csv')


# In[4]:


#ーーーーー 保存先のフォルダの確認 ーーーーー

# %pwd


# ### 例

# 例のファイルを使う場合。

# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/Haruyama-KobeU/Py4Basics/master/data/data_for_mark.csv')
df


# ## 列の内容の確認

# ステップ１：`for`ループによる列のラベルの表示

# In[6]:


col = df.columns
col


# In[7]:


for c in col:
    print(c)


# ステップ２：それぞれの列の要素の種類を表示方法

# In[8]:


df['区分'].unique()


# スッテプ３：スッテプ１と２を同時に

# In[9]:


for c in col[1:]:      # Noの列を除外
    x = df[c].unique()
    print(c, x)


# ## `DataFrame`の作成

# In[10]:


df_other = df.query('区分 == "全学共通授業科目" or 区分 == "高度教養科目"')
df_econ = df.query('区分 == "専門科目"')


# In[11]:


len(df),len(df_other),len(df_econ)


# 全角文字を使わない方法

# In[12]:


kubun = df.loc[:,col[1]]

kubun_arr = kubun.unique()
kubun_arr


# In[13]:


other = ( ( kubun == kubun_arr[0] ) | ( kubun == kubun_arr[1] ) )
econ = ( kubun == kubun_arr[2] )


# In[14]:


df_other = df.loc[other,:]
df_econ = df.loc[econ,:]


# In[15]:


len(df),len(df_other),len(df_econ)


# ## 全科目

# ### `f-string`

# `f-string`を使うと文字列の`{}`の中の変数を評価して表示することが可能となる。

# In[16]:


x='春山'
print(f'私は{x}ゼミに所属しています。')


# In[17]:


l = [1,2,3]
print(f'合計は{sum(l)}です。')


# ### 優・秀・良・可・不可などの数

# #### 簡単な方法

# `value_counts()`を使うと簡単になる。

# In[18]:


df.loc[:,'評語'].value_counts()


# #### 表示を整理したい場合

# 評語の種類

# In[19]:


m = df.loc[:,'評語'].unique()
m


# In[20]:


mark = [m[2],m[3],m[4],m[1],m[0],m[-1],m[-2],m[-3]]
mark


# In[21]:


for m in mark:
    print(m)

print('--------\n合計')


# In[22]:


lst = []

for m in mark:
    no = len(df.query('評語 == @m'))
    lst.append(no)

lst


# In[23]:


lst = []

for m in mark:
    no = len(df.query('評語 == @m'))
    lst.append(no)
    print(m, no)

print('--------\n合計',sum(lst))


# ### 優・秀・良・可・不可などの％

# #### 簡単な方法

# In[24]:


(100 * df.loc[:,'評語'].value_counts() / len(df.loc[:,'評語']) ).round(1)


# #### 表示を整理したい場合

# ％だけを表示する。

# In[25]:


lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)

lst


# ％と評語も表示する。

# In[26]:


lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, f': {percent}')
    
print(f'-----------\n合計: {sum(lst)}')


# 小数点の表示を調整する。

# In[27]:


lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, f': {percent:.1f}')   # 小数点第ー位まで表示
    
print(f'-----------\n合計: {sum(lst):.0f}')  # 小数点は表示しない


# 上のコードでは`f-string`を使った。その代わりに`format()`を使うことも可能。

# In[28]:


lst = []

for m in mark:
    percent = 100 * len(df.query('評語 == @m')) / len(df)
    lst.append(percent)
    print(m, '{:.1f}'.format(percent))
    
print('----------\n合計 {:.0f}'.format(sum(lst)))


# ## 全学共通授業科目

# ### 数

# In[29]:


df_other.loc[:,'評語'].value_counts()


# In[30]:


lst = []

for m in mark:
    no = len(df_other.query('評語 == @m'))
    lst.append(no)
    print(m,no)
    
print(f'--------\n合計 {sum(lst)}')


# ### 割合

# In[31]:


(100 * df_other.loc[:,'評語'].value_counts() / len(df_other.loc[:,'評語']) ).round(1)


# In[32]:


lst = []

for m in mark:
    percent = 100 * len(df_other.query('評語 == @m')) / len(df_other)
    lst.append(percent)
    print(m, f'{percent:.1f}')
    
print(f'-----------\n合計 {sum(lst):.0f}')


# ## 専門科目

# In[33]:


df_econ.loc[:,'評語'].value_counts()


# In[34]:


lst = []

for m in mark:
    no = len(df_econ.query('評語 == @m'))
    lst.append(no)
    print(m,no)
    
print(f'--------\n合計 {sum(lst)}')


# In[35]:


(100 * df_econ.loc[:,'評語'].value_counts() / len(df_econ.loc[:,'評語']) ).round(1)


# In[36]:


lst = []

for m in mark:
    percent = 100 * len(df_econ.query('評語 == @m')) / len(df_econ)
    lst.append(percent)
    print(m, f'{percent:.1f}')
    
print(f'--------\n合計 {sum(lst):.0f}')


# ## GPAの推移

# ### 「科目GP」で記号がある行の削除

# 科目GPの要素の種類

# In[37]:


df.loc[:,'科目GP'].unique()


# 全て文字列となっているので，`-`と`*`の記号が含まれない行だけから構成される`DataFrame`を作成する。
# 
# 後で「科目GP」のデータ型を変更する際に警告がでないようにメソッド`.copy()`を使い`DataFrame`のコピーを作成する。

# In[38]:


gpa = df.query("科目GP not in ['-', '*']").copy()
gpa.head()


# In[39]:


gpa['科目GP'].unique()


# データ型は`object`（文字列）のままである。
# 
# 属性`.dtypes`を使って確認することもできる。

# In[40]:


gpa['科目GP'].dtypes


# `O`は`object`。

# ### 「科目GP」を浮動小数点に変更

# 科目GPは`object`（文字列）となっているので，メソッド`astype()`を使って`float`に変換する。

# In[41]:


gpa['科目GP'] = gpa['科目GP'].astype(float)


# In[42]:


gpa.dtypes


# ### 図示：毎年

# #### `groupby`

# `groupby`は`DataFrame`や`Series`をグループ化し，グループ内の計算を簡単に行うことができる便利なメソッドである。「修得年度」でグループ化し，平均を計算する。

# In[43]:


gpa_grouped = gpa.groupby('修得年度')
gpa_mean = gpa_grouped.mean()
gpa_mean


# * `gpa_grouped`は`DetaFrame`ではないので注意しよう。
# * `gpa_grouped`には`median()`，`std()`，`var()`，`min()`，`max()`などが使える。その他の属性・メソッドを確認。

# In[44]:


see(gpa_grouped)


# #### 図示

# 方法１：横軸に文字列を使う

# `gpa_mean.index`を確認する。

# In[45]:


gpa_mean.index


# これを使い横軸に使う文字列を作成する。

# In[46]:


yr = [str(i) for i in gpa_mean.index]
yr


# In[47]:


plt.plot(yr,gpa_mean['科目GP'], 'o-')
pass


# 方法２：`plt.xticks()`を使って

# 横軸を調整するために，`gpa_mean`のインデックスを確認。

# In[48]:


yr = gpa_mean.index
yr


# `xticks()`を使って横軸の表示を指定する。

# In[49]:


plt.plot('科目GP', 'o-', data=gpa_mean)
plt.xticks(yr)
pass


# ### 図示：前期後期毎

# #### 「修得学期」の英語化

# 「修得学期」の要素が日本語だと警告が出るため，メソッド`.replace()`を使って英語に変換する。`replace()`の引数は辞書で指定する。
# * キー：変更する対象の数値や文字列
# * 値：変更後の数値や文字列

# In[50]:


gpa['修得学期'] = gpa['修得学期'].replace({'前期':'1','後期':'2'})
gpa.head()


# #### `groupby`

# 「修得年度」と「修得学期」でグループ化する。

# In[51]:


gpa_grouped = gpa.groupby(['修得年度','修得学期'])
gpa_mean = gpa_grouped.mean()
gpa_mean


# 図示する際エラーをがでないようにインデックスのラベルを削除する。

# In[52]:


gpa_mean.index.names = (None,None)
gpa_mean


# #### 図示

# 横軸に文字列を使う。まず`gpa_mean`のインデックスを確認する。

# In[53]:


gpa_mean.index


# これを使い横軸に使う文字列を作成する。

# In[54]:


yr_half = [str(i[0])+'-'+str(i[1]) for i in gpa_mean.index]
yr_half


# In[55]:


plt.plot(yr_half, gpa_mean['科目GP'])
pass

