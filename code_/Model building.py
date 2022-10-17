# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/11 20:24
@Auth ： Xiaoyao.Jia
@File ：Model building.py
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

classfiers = {

        'CatBoostClassifier': CatBoostClassifier(random_state=random_seed,verbose=0),
        'GradientBoostingClassifier':GradientBoostingClassifier(random_state=random_seed),
        'RandomForestClassifier':RandomForestClassifier(random_state=random_seed),
        'XGBClassifier': XGBClassifier(random_state=random_seed)
        ,'ExtraTreesClassifier':ExtraTreesClassifier(random_state=random_seed),
        'LogisticRegression': LogisticRegression(),
        'SVC':SVC()
        ,'KNeighborsClassifier':KNeighborsClassifier(),
         'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(),
         'DecisionTreeClassifier':DecisionTreeClassifier()
        ,'ExtraTreeClassifier':ExtraTreeClassifier(random_state=random_seed),
          'MLPClassifier':MLPClassifier()
        ,'AdaBoostClassifier':AdaBoostClassifier(random_state=random_seed),

         'LGBMClassifier': LGBMClassifier(random_state=random_seed) ,
}
train_pd_X = pd.read_csv('../data_/cache/X_sel_X.csv', encoding='utf-8')
train_pd_V = pd.read_csv('../data_/cache/X_sel_V.csv', encoding='utf-8')
train_pd_M = pd.read_csv('../data_/cache/X_sel_mic.csv', encoding='utf-8')
y = pd.read_csv('../data_/cache/y.csv', encoding='utf-8')

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc','f1':'f1','precision':'precision','recall':'recall'}
splits_num = 5
skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)
models_data_1 = {'Accuracy': {},'Accuracy_std': {}, 'AUC': {},'AUC_std': {}, 'Precision': {},'Precision_std': {}, 'Recall': {},'Recall_std': {},'F1-score':{},'F1-score_std':{}}
for cls_name, cls in classfiers.items():
    print(cls_name)
    results = pd.DataFrame(cross_validate(cls, train_pd_X, y, cv=skf, scoring=list(scoring.values()),return_train_score=False))
    models_data_1['Accuracy'][cls_name] = results.mean()['test_accuracy']
    models_data_1['Accuracy_std'][cls_name] = results.std()['test_accuracy']
    models_data_1['AUC'][cls_name] = results.mean()['test_roc_auc']
    models_data_1['AUC_std'][cls_name] = results.std()['test_roc_auc']
    models_data_1['Recall'][cls_name] = results.mean()['test_recall']
    models_data_1['Recall_std'][cls_name] = results.std()['test_recall']
    models_data_1['Precision'][cls_name] = results.mean()['test_precision']
    models_data_1['Precision_std'][cls_name] = results.std()['test_precision']
    models_data_1['F1-score'][cls_name] = results.mean()['test_f1']
    models_data_1['F1-score_std'][cls_name] = results.std()['test_f1']
models_df_1 = pd.DataFrame(models_data_1).sort_values(by='Accuracy',ascending=False)
print(models_df_1)
models_df_1.to_excel('../data_/cache/re_X.excel',index=False)

models_data_2 = {'Accuracy': {},'Accuracy_std': {}, 'AUC': {},'AUC_std': {}, 'Precision': {},'Precision_std': {}, 'Recall': {},'Recall_std': {},'F1-score':{},'F1-score_std':{}}
for cls_name, cls in classfiers.items():
    print(cls_name)
    results = pd.DataFrame(cross_validate(cls, train_pd_V, y, cv=skf, scoring=list(scoring.values()),return_train_score=False))
    models_data_2['Accuracy'][cls_name] = results.mean()['test_accuracy']
    models_data_2['Accuracy_std'][cls_name] = results.std()['test_accuracy']
    models_data_2['AUC'][cls_name] = results.mean()['test_roc_auc']
    models_data_2['AUC_std'][cls_name] = results.std()['test_roc_auc']
    models_data_2['Recall'][cls_name] = results.mean()['test_recall']
    models_data_2['Recall_std'][cls_name] = results.std()['test_recall']
    models_data_2['Precision'][cls_name] = results.mean()['test_precision']
    models_data_2['Precision_std'][cls_name] = results.std()['test_precision']
    models_data_2['F1-score'][cls_name] = results.mean()['test_f1']
    models_data_2['F1-score_std'][cls_name] = results.std()['test_f1']
models_df_2 = pd.DataFrame(models_data_2).sort_values(by='Accuracy',ascending=False)
print(models_df_2)
models_df_2.to_excel('../data_/cache/re_V.excel',index=False)

models_data_3 = {'Accuracy': {},'Accuracy_std': {}, 'AUC': {},'AUC_std': {}, 'Precision': {},'Precision_std': {}, 'Recall': {},'Recall_std': {},'F1-score':{},'F1-score_std':{}}
for cls_name, cls in classfiers.items():
    print(cls_name)
    results = pd.DataFrame(cross_validate(cls, train_pd_V, y, cv=skf, scoring=list(scoring.values()),return_train_score=False))
    models_data_3['Accuracy'][cls_name] = results.mean()['test_accuracy']
    models_data_3['Accuracy_std'][cls_name] = results.std()['test_accuracy']
    models_data_3['AUC'][cls_name] = results.mean()['test_roc_auc']
    models_data_3['AUC_std'][cls_name] = results.std()['test_roc_auc']
    models_data_3['Recall'][cls_name] = results.mean()['test_recall']
    models_data_3['Recall_std'][cls_name] = results.std()['test_recall']
    models_data_3['Precision'][cls_name] = results.mean()['test_precision']
    models_data_3['Precision_std'][cls_name] = results.std()['test_precision']
    models_data_3['F1-score'][cls_name] = results.mean()['test_f1']
    models_data_3['F1-score_std'][cls_name] = results.std()['test_f1']
models_df_3 = pd.DataFrame(models_data_2).sort_values(by='Accuracy',ascending=False)
print(models_df_3)
models_df_3.to_excel('../data_/cache/re_M.excel',index=False)

'''roc'''
result_pd = pd.DataFrame()
cls_nameList = []
accuracys=[]
precisions=[]
recalls=[]
F1s=[]
AUCs=[]
MMCs = []

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
    for k, (train_index, test_index) in enumerate(skf.split(X=train_pd_V, y=y)):
        X_train,X_test, y_train, y_test = train_pd_V[train_index], train_pd_V[test_index], y[train_index],y[test_index]

        # 这里可以加入不平衡处理的代码
        # X_resampled, y_resampled = combine.SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
        # cls.fit(X_resampled, y_resampled)
        cls.fit(X_train, y_train)
        viz = RocCurveDisplay.from_estimator(cls,X_test,y_test,
            name="ROC fold {}".format(k),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        y_pred = cls.predict(X_test)
        total_accuracy += accuracy_score(y_test, y_pred)
        total_precision += precision_score(y_test, y_pred)
        total_recall += recall_score(y_test, y_pred)
        total_f1 += f1_score(y_test, y_pred)
        total_auc += roc_auc_score(y_test, y_pred)
        total_mmc += matthews_corrcoef(y_test, y_pred)
        explainercat = shap.TreeExplainer(cls)

    cls_nameList.append(cls_name)
    accuracys.append(total_accuracy/splits_num)
    precisions.append(total_precision/splits_num)
    recalls.append(total_recall/splits_num)
    F1s.append(total_f1/splits_num)
    AUCs.append(total_auc/splits_num)
    MMCs.append(total_mmc/splits_num)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()

'''折线可视化'''
#df为性能平均值，方差这两列的结果表
df=pd.read_excel()
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
fig = plt.figure(figsize=(10, 10))
iters = list(df.index)

color = palette(0)  # 算法1颜色
ax = fig.add_subplot(1, 1,  1)
avg = np.array(df.iloc[:,0])
std = np.array(df.iloc[:,1])
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
ax.plot(iters, avg, color=color, label="Feature selection based on Wrapper", linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

color = palette(1)
avg = np.array(df.iloc[:,2])
std = np.array(df.iloc[:,3])
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax.plot(iters, avg, color=color, label="Feature selection based on MIC", linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

color = palette(2)
avg = np.array(df.iloc[:,4])
std = np.array(df.iloc[:,5])
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax.plot(iters, avg, color=color, label="Feature selection based on Variance Filtering", linewidth=3.0)
ax.fill_between(iters, r1, r2, color=color, alpha=0.2)


ax.legend(loc='lower left', prop=font1)
ax.set_xlabel('Names of different algorithms', fontsize=22)
plt.xticks(fontsize=15)
ax.set_ylabel('F1', fontsize=22)
plt.show()



