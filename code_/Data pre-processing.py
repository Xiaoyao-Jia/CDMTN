# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/11 17:06
@Auth ： Xiaoyao.Jia
@File ：Data pre-processing.py
@IDE ：PyCharm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler,Normalizer
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from feature_selector import FeatureSelector
from scipy.stats import mstats
from imblearn import over_sampling,under_sampling,combine
# from deepforest.cascade import CascadeForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imbalanced_ensemble.ensemble.under_sampling import BalanceCascadeClassifier, UnderBaggingClassifier
from imbalanced_ensemble.ensemble.over_sampling import OverBoostClassifier, SMOTEBoostClassifier, KmeansSMOTEBoostClassifier, OverBaggingClassifier,SMOTEBaggingClassifier
from imbalanced_ensemble.ensemble.reweighting import AdaCostClassifier,AdaUBoostClassifier,AsymBoostClassifier
from imbalanced_ensemble.ensemble.compatible import CompatibleAdaBoostClassifier, CompatibleBaggingClassifier
from self_paced_ensemble import SelfPacedEnsembleClassifier
import shap
import seaborn as sns
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
random_seed = 0


import matplotlib.patches as mpatches
from matplotlib import pyplot

train_pd = pd.read_csv('../data_/thyroid_clean.csv', encoding='utf-8')
# 特征列
feature_cols = [col for col in train_pd.columns if col not in ['id', 'mal']]#去除id 和mal这两列的特征名
X = train_pd[feature_cols]
y = train_pd['mal']
# '''ro前后对比'''
# palette = pyplot.get_cmap('Set1')
# sns.scatterplot(data=train_pd, x="FT3", y="FT4", hue="mal", palette='Set2')
# plt.show()
# ''''''
# palette = pyplot.get_cmap('Set1')

for i in X.columns:
    X[i]= pd.Series(mstats.winsorize(X[i], limits=[0.01, 0.01]))

X = RobustScaler().fit_transform(X)#0.780073 CatB
X = pd.DataFrame(X, columns=feature_cols)
# df=pd.concat([X,y],axis=1)
# '''ro前后对比'''
# palette = pyplot.get_cmap('Set1')
# sns.scatterplot(data=df, x="FT3", y="FT4", hue="mal", palette='Set2')
# plt.show()
# ''''''

# 定义特征构造方法，构造特征
epsilon = 1e-5
func_dict={
    'add': lambda x, y : x+y,
    'mins': lambda x, y: x-y,
    'multi': lambda x, y: x*y,
    'div': lambda x,y: x/(y+epsilon)
}

# 定义特征构造的函数
def auto_make_feature(train_data, func_dict, col_list):
    train_data = train_data.copy()
    for col_i in col_list:
        for col_j in col_list:
            for func_name, func in func_dict.items():
                    func_features = func(train_data[col_i], train_data[col_j])
                    col_func_features = '-'.join([col_i,func_name,col_j])#通过“-”字符链接，构造衍生特征的特征名字
                    train_data[col_func_features]=func_features
    return train_data

X = auto_make_feature(X, func_dict, col_list=X.columns)
# 存储特征名称
all_feature_name = X.columns.values.tolist()
from sklearn.feature_selection import SelectFromModel
# 使用树模型选择特征 方法N
clf = XGBClassifier()
# clf = LGBMClassifier()
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_sel_X = pd.DataFrame(model.transform(X))
X_sel_X.to_csv('../data_/cache/X_sel_X.csv',index=False)


X_sel_V = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
pd.DataFrame(X_sel_V).to_csv('../data_/cache/X_sel_V.csv',index=False)


from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest
result = MIC(X_sel_V,y)
k = result.shape[0] - sum(result <= 0)
X_sel_mic = SelectKBest(MIC, k=k).fit_transform(X_sel_V, y)
pd.DataFrame(X_sel_mic).to_csv('../data_/cache/X_sel_mic.csv',index=False)

y.to_csv('../data_/cache/y.csv',index=False)
