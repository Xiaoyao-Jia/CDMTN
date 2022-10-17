# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/12 11:42
@Auth ： Xiaoyao.Jia
@File ：shap.py
@IDE ：PyCharm
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler,Normalizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
'''matthews_corrcoef MCC  马修斯相关系数（Matthews correlation coefficient）'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef,RocCurveDisplay,auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 分类器汇总
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn import over_sampling,under_sampling,combine
# from deepforest.cascade import CascadeForestClassifier
# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.ensemble import RUSBoostClassifier
# from imblearn.ensemble import EasyEnsembleClassifier
# from imbalanced_ensemble.ensemble.under_sampling import BalanceCascadeClassifier, UnderBaggingClassifier
# from imbalanced_ensemble.ensemble.over_sampling import OverBoostClassifier, SMOTEBoostClassifier, KmeansSMOTEBoostClassifier, OverBaggingClassifier,SMOTEBaggingClassifier
# from imbalanced_ensemble.ensemble.reweighting import AdaCostClassifier,AdaUBoostClassifier,AsymBoostClassifier
# from imbalanced_ensemble.ensemble.compatible import CompatibleAdaBoostClassifier, CompatibleBaggingClassifier
# from self_paced_ensemble import SelfPacedEnsembleClassifier
import shap
from scipy.stats import mstats
# from feature_selector import FeatureSelector

random_seed = 0
train_pd = pd.read_csv('../data_/thyroid_clean.csv', encoding='utf-8')
feature_cols = [col for col in train_pd.columns if col not in ['id', 'mal']]
X = train_pd[feature_cols]
y = train_pd['mal']
for i in X.columns:
    X[i]= pd.Series(mstats.winsorize(X[i], limits=[0.01, 0.01]))

X = RobustScaler().fit_transform(X)#0.780073 CatB
X = pd.DataFrame(X, columns=feature_cols)

classfiers = {

        'CatBoostClassifier': CatBoostClassifier(random_state=random_seed,verbose=0),}
result_pd = pd.DataFrame()
cls_nameList = []
accuracys=[]
precisions=[]
recalls=[]
F1s=[]
AUCs=[]
MMCs = []
splits_num=5
skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

for cls_name, cls in classfiers.items():
    print("start training:", cls_name)
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_auc = 0.0
    total_mmc = 0.0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
        X_train,X_test, y_train, y_test = X[train_index], X[test_index], y[train_index],y[test_index]

        # 这里可以加入不平衡处理的代码
        # X_resampled, y_resampled = combine.SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
        # cls.fit(X_resampled, y_resampled)
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        total_accuracy += accuracy_score(y_test, y_pred)
        total_precision += precision_score(y_test, y_pred)
        total_recall += recall_score(y_test, y_pred)
        total_f1 += f1_score(y_test, y_pred)
        total_auc += roc_auc_score(y_test, y_pred)
        total_mmc += matthews_corrcoef(y_test, y_pred)
        explainer = shap.TreeExplainer(cls)
        shap_values_train = explainer.shap_values(X_train)
        plt.figure(figsize=(10, 10), dpi=1000)
        a = shap.summary_plot(shap_values_train, X_train, show=False, max_display=10,plot_type="bar",)  # cmap=plt.get_cmap('autumn')
        plt.savefig('shap总.png')
        plt.show()

        clustering = shap.utils.hclust(X_train, y_train)
        shap.initjs()
        plt.figure(figsize=(10, 10), dpi=1000)
        shap_values_obj = explainer(X_train)
        shap.plots.bar(shap_values_obj,
                       clustering=clustering, show=False
                       #                clustering_cutoff=0.5
                       )
        plt.savefig('shaptiaohong.png')
        plt.show()

        explainer = shap.Explainer(cls.predict, X_test.iloc[:10, :])
        shap_values = explainer(X_test.iloc[:10, :])

        shap.initjs()
        shap.plots.waterfall(shap_values[4], show=False)  # 1
        plt.savefig('shap类别1.png')
        plt.show()

        shap.initjs()
        shap.plots.waterfall(shap_values[6], show=False)  # 0
        plt.savefig('shap类别0.png')
        plt.show()

