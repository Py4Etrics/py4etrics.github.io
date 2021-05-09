#!/usr/bin/env python
# coding: utf-8

# # Gapminder

# If you come here without expecting Japanese, please click [Google translated version](https://translate.google.com/translate?hl=&sl=ja&tl=en&u=https%3A%2F%2Fpy4etrics.github.io%2FGapminder.html) in English or the language of your choice.
# 
# ---

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gapminder import gapminder
from see import see


# [Gapminder](https://www.gapminder.org)とは世界規模で見た経済格差をデータで探る有名なサイトであり、一見の価値があるサイトである。そのサイトで使われているデータを整理してパッケージにまとめたのが`gapminder`である。
# 
# ````{note}
# MacではTerminal、WindowsではGit Bashを使い、次のコマンドで`gapminder`をインストールできる。
# ```
# pip install gapminder
# ```
# ````
# 
# ここでは`gapminder`に含まれるデータを使い`pandas`の`groupby`という`DataFrame`のメソッドと`Multi-index`（階層型インデックス）の使い方の例を紹介する。両方ともデータをグループ化して扱う場合に非常に重宝するので、覚えておいて損はしないだろう。

# ## データ

# In[2]:


df = gapminder
df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# **含まれる国名**

# In[6]:


countries = df.loc[:,'country'].unique()
countries


# **国数**

# In[7]:


len(countries)


# **`continent`の内訳**

# In[8]:


# continentのリスト

continent_list = np.sort(df.loc[:,'continent'].unique()).tolist()
continent_list


# In[9]:


# 国数を入れる空のリスト
num_countries = []

for i in continent_list:
    # 条件
    con = df.loc[:,'continent'] == i
    #　条件が満たされる行を抽出
    series_countries = df.loc[con,'country']
    # 重複をなくす
    series_countries_unique = series_countries.unique()
    # 国数を数える
    num = len(series_countries_unique)
    # 上のリストに追加
    num_countries.append(num)


# In[10]:


for name, num in zip(continent_list,num_countries):
    print(f'{name}: {num}')


# In[11]:


sum(num_countries)


# ## groupby()

# In[12]:


df_group = df.groupby('continent')


# In[13]:


see(df_group)


# ### continentの内訳（again）

# In[14]:


# Seriesを返し，要素はarray

country_names = df_group['country'].unique()
country_names


# In[15]:


N = len(country_names)

for i in range(N):
    t0 = country_names[i]
    t1 = len(t0)
    print(country_names.index[i],':',t1)


# ### 統計量

# In[16]:


three_vars=['lifeExp','pop','gdpPercap']


# #### データ数

# In[17]:


df_group.count()


# In[18]:


df_group.size().plot(kind='bar')
pass


# #### 平均

# In[19]:


df_group[three_vars].mean()


# In[20]:


df_group.mean().plot(kind='scatter', x='gdpPercap', y='lifeExp')
pass


# #### 標準偏差

# In[21]:


df_group[three_vars].std()


# #### 最大値

# In[22]:


df_group.max()


# #### 最小値

# In[23]:


df_group.min()


# **基本的統計**

# In[24]:


df_group[three_vars].describe()


# ### groupby.agg()

# `agg()`を使うとよりメソッドだけではなく，他の関数も使える。
# 
# `()`の中に関数を入れる。

# In[25]:


df_group.agg(np.mean)


# In[26]:


df_group[three_vars].agg([np.max, np.min, np.mean])


# In[27]:


# 自作の関数もOK

func = lambda x : (np.max(x)-np.min(x))/np.mean(x)

df_group['lifeExp','pop','gdpPercap'].agg(func)


# ### 図

# **`continent`平均**

# In[28]:


df_lifeExp_continent = df_group['lifeExp'].mean()


# In[29]:


df_lifeExp_continent.plot(kind='bar')
pass


# #### クロス・セクション

# In[30]:


df_mean = df_group.mean()
df_mean['ln_pop'] = np.log(df_mean['pop'])
df_mean['ln_gdpPercap'] = np.log(df_mean['gdpPercap'])
df_mean['lifeExp_10'] = df_mean['lifeExp']/10


# In[31]:


df_mean[['lifeExp_10', 'ln_gdpPercap']].plot(kind='bar')
pass


# ### 複数階層の`groupby()`

# `continent`別の平均時系列を考えるときに有用。

# In[32]:


df_group2 = df.groupby(['continent','year'])


# In[33]:


df_group2.mean().head()


# In[34]:


# lifeExpの列だけを選択した後，行はyear列はcontinentになるDataFrameに変換

df_lifeExp_group = df_group2.mean().loc[:,'lifeExp'].unstack(level=0)


# In[35]:


df_lifeExp_group.plot()
plt.ylabel('Average Life Expectancy')
pass


# **世界平均との比較**

# In[36]:


df_group_year = df.groupby('year')


# In[37]:


world_lifeExp = df_group_year.mean()['lifeExp'].values.reshape(1,12).T


# In[38]:


df_lifeExp_diff = df_lifeExp_group - world_lifeExp
df_lifeExp_diff.plot()
pass


# ## Multi-index

# In[39]:


# sort_index()がないとWarningがでる場合がある(順番で並ぶとPythonが変数を探しやすくなる)

dfm = df.set_index(['continent','country','year']).sort_index()


# In[40]:


dfm.head()


# `continent`, `country`, `year`の３つがインデックス！

# ### 統計量

# #### データ数

# In[41]:


dfm.count(level=0)


# #### 平均の計算

# In[42]:


dfm.mean(level='continent')

# dfm.mean(level=0) も同じ


# #### 標準偏差

# In[43]:


dfm.std(level='continent')


# #### 最大値・最小値

# In[44]:


dfm.max(level='continent')


# In[45]:


dfm.min(level='continent')


# ### 図

# **図（５カ国の時系列）**

# In[46]:


dfm_2 = dfm.droplevel(level=0,axis=0)


# In[47]:


countries = ['Japan', 'United Kingdom', 'United States', 'China', 'Thailand']

dfm_2.loc[(countries),'lifeExp'].unstack(level=0).plot()
pass


# **`lifeExp`の世界平均との差**

# In[48]:


df_lifeExp_mi = pd.DataFrame()

for i in continent_list:
    temp = dfm.loc[(i,),'lifeExp'].unstack(level=0).mean(axis=1)
    df_lifeExp_mi[i] = temp


# In[49]:


df_lifeExp_mi.plot()
plt.title('Average Life Expectancy')
pass


# In[50]:


world_lifeExp_mi = dfm_2['lifeExp'].unstack(level=0).mean(axis=1).values.reshape(12,1)


# In[51]:


df_lifeExp_diff_mi = df_lifeExp_mi - world_lifeExp_mi


# In[52]:


df_lifeExp_diff_mi.plot()
pass

