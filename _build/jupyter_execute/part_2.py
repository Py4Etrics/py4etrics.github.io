#!/usr/bin/env python
# coding: utf-8

# <!--# ２. `Python`を使った計量経済分析-->
# # 目次と内容
# 
# ```{margin}
# <div name="html-admonition">
# Do you want to read in a differnt language? Open the 
# <input type="button" onclick="location.href='https://translate.google.com/translate?hl=&sl=ja&tl=en&u='+window.location;" value="Google translated version" style="color:#ffffff;background-color:#008080;" onmouseover="this.style.background='#99ccff'" onmouseout="this.style.background='#008080'"/>
# in English or the language of your choice.
# </div>
# ```
# 
# 1. {doc}`7_Review_of_Statistics` 
#    - 平均、分散、共分散などの計算方法の復習
# 1. {doc}`list_of_terms`
#    - 計量経済学用語のリスト
# 1. {doc}`8_Simple_Regression`
#    - 最も簡単な計量経済分析
# 1. {doc}`9_Multiple_Regression`
#    - 複数の説明変数がある場合であり，ガウス・マルコフ定理について説明する。
# 1. {doc}`10_Residuals`
#    - 残差を使い回帰分析に必要な仮定が成立しているかを図示でチェックする。
# 1. {doc}`11_Inference`
#    - 仮説の検定など
# 1. {doc}`12_Asymptotics`
#    - 標本サイズが大きい場合の分析
# 1. {doc}`13_Dummies`
#    - ダミー変数やカテゴリー変数を扱う場合を考える
# 1. {doc}`14_Hetero`
#    - 残差の均一分散の過程が成立しない場合の対処方法
# 1. {doc}`15_Pooling`
#    - データに時系列とクロスセクションの情報がある場合の分析
# 1. {doc}`16_linearmodels`
#    - パネル・データと操作変数法を扱うためのパッケージについての説明
# 1. {doc}`17_Panel`
#    - パネル・データの利点を最大限に利用する手法
# 1. {doc}`18_Zero_Conditional_Mean`
#    - ガウス・マルコフ定理の１つの仮定である「残差の条件付き期待値がゼロ」が成立しない場合のバイアスをシミュレーションで確認する
# 1. {doc}`19_IV2SLS`
#    - 残差の条件付き期待値がゼロでない場合の対処方法
# 1. {doc}`20_LogitProbit`
#    - ロジット・モデルとプロビット・モデルを使った分析方法
# 1. {doc}`21_TruncregTobitHeckit`
#    - 切断回帰、Tobit、Heckitモデルを考える
