# Gapminder

## データ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gapminder import gapminder

df = gapminder
df.head()

df.tail()

df.info()

df.describe()

**含まれる国名**

countries = df.loc[:,'country'].unique()
countries

**国数**

len(countries)

**`continent`の内訳**

# continentのリスト

continent_list = np.sort(df.loc[:,'continent'].unique()).tolist()
continent_list

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

for name, num in zip(continent_list,num_countries):
    print(f'{name}: {num}')

sum(num_countries)

## groupby()

df_group = df.groupby('continent')

from see import *
see(df_group)

### continentの内訳（again）

# Seriesを返す

country_names = df_group['country'].unique()
country_names

N = len(country_names)

for i in range(N):
    t0 = country_names[i]
    t1 = len(t0)
    print(country_names.index[i],':',t1)

### 統計量

three_vars=['lifeExp','pop','gdpPercap']

#### データ数

df_group.count()

df_group.size().plot(kind='bar')
pass

#### 平均

df_group[three_vars].mean()

df_group.mean().plot(kind='scatter', x='gdpPercap', y='lifeExp')
pass

#### 標準偏差

df_group[three_vars].std()

#### 最大値

df_group.max()

#### 最小値

df_group.min()

**基本的統計**

df_group[three_vars].describe()

### groupby.agg()

`agg()`を使うとよりメソッドだけではなく，他の関数も使える。

`()`の中に関数を入れる。

df_group.agg(np.mean)

df_group[three_vars].agg([np.max, np.min, np.mean])

# 自作の関数もOK

func = lambda x : (np.max(x)-np.min(x))/np.mean(x)

df_group['lifeExp','pop','gdpPercap'].agg(func)

### 図

**`continent`平均**

df_lifeExp_continent = df_group['lifeExp'].mean()

df_lifeExp_continent.plot(kind='bar')
pass

#### クロス・セクション

df_mean = df_group.mean()
df_mean['ln_pop'] = np.log(df_mean['pop'])
df_mean['ln_gdpPercap'] = np.log(df_mean['gdpPercap'])
df_mean['lifeExp_10'] = df_mean['lifeExp']/10

df_mean[['lifeExp_10', 'ln_gdpPercap']].plot(kind='bar')
pass

### 複数階層の`groupby()`

`continent`別の平均時系列を考えるときに有用。

df_group2 = df.groupby(['continent','year'])

df_group2.mean().head()

# lifeExpの列だけを選択した後，行はyear列はcontinentになるDataFrameに変換

df_lifeExp_group = df_group2.mean().loc[:,'lifeExp'].unstack(level=0)

df_lifeExp_group.plot()
plt.ylabel('Average Life Expectancy')
pass

**世界平均との比較**

df_group_year = df.groupby('year')

world_lifeExp = df_group_year.mean()['lifeExp'].values.reshape(1,12).T

df_lifeExp_diff = df_lifeExp_group - world_lifeExp
df_lifeExp_diff.plot()
pass

## Multi-index

# sort_index()がないとWarningがでる場合がある(順番で並ぶとPythonが変数を探しやすくなる)

dfm = df.set_index(['continent','country','year']).sort_index()

dfm.head()

`continent`, `country`, `year`の３つがインデックス！

### 統計量

#### データ数

dfm.count(level=0)

#### 平均の計算

dfm.mean(level='continent')

# dfm.mean(level=0) も同じ

#### 標準偏差

dfm.std(level='continent')

#### 最大値・最小値

dfm.max(level='continent')

dfm.min(level='continent')

### 図

**図（５カ国の時系列）**

dfm_2 = dfm.droplevel(level=0,axis=0)

countries = ['Japan', 'United Kingdom', 'United States', 'China', 'Thailand']

dfm_2.loc[(countries),'lifeExp'].unstack(level=0).plot()
pass

**`lifeExp`の世界平均との差**

df_lifeExp_mi = pd.DataFrame()

for i in continent_list:
    temp = dfm.loc[(i,),'lifeExp'].unstack(level=0).mean(axis=1)
    df_lifeExp_mi[i] = temp

df_lifeExp_mi.plot()
plt.title('Average Life Expectancy')
pass

world_lifeExp_mi = dfm_2['lifeExp'].unstack(level=0).mean(axis=1).values.reshape(12,1)

df_lifeExp_diff_mi = df_lifeExp_mi - world_lifeExp_mi

df_lifeExp_diff_mi.plot()
pass