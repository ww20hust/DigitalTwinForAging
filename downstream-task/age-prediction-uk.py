import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Lasso
import xgboost as xgb
import Utils_ehr
from Utils_ehr import *
# 忽略特定的警告
warnings.filterwarnings("ignore")
import importlib
importlib.reload(Utils_ehr)
import joblib

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from scipy.stats import pearsonr
import scipy.io as sio
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, cross_val_predict

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

meta_data = pd.read_parquet('/data2404/ww/Project/卵巢早衰课题/提取ukb的数据/1-feature_selection_meta_data.parquet')
# df_body_age_sub = pd.read_parquet('0309_df_body_age.parquet')
df_body_age_sub = pd.read_parquet('0312_df_body_age_missing50.parquet')
df_body_age_sub_columns = list(df_body_age_sub.columns)
df_body_age_sub_columns.remove('eid')

with open("/data2404/ww/Project/卵巢早衰课题/organaging/organ-index.json",'r',encoding='utf-8') as load_f:
    organ_dict = json.load(load_f)
organ_dict_index = list(set([index for item in organ_dict for index in organ_dict[item]]))

meta_data = pd.read_parquet('/data2404/ww/Project/卵巢早衰课题/提取ukb的数据/1-feature_selection_meta_data.parquet')
meta_data = meta_data.rename(columns={
    'p3166_i0': 'Date of Blood-0',
    'p31': 'sex'
})
meta_data_sub = meta_data[['eid','sex','Date of Blood-0','Date of birth']].copy()
body_func_feature_sex_age = pd.merge(df_body_age_sub,meta_data_sub,on = 'eid',how = 'inner')

body_func_feature_sex_age['missing_count'] = body_func_feature_sex_age.isnull().sum(axis=1)
plt.hist(body_func_feature_sex_age['missing_count'], bins=range(body_func_feature_sex_age['missing_count'].max() + 2), rwidth=0.8)
plt.xlabel('missing num')
plt.ylabel('sample num')
plt.show()
# body_func_feature_sex_age = body_func_feature_sex_age[body_func_feature_sex_age['missing_count']<=20]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
body_func_feature_sex_age_scaled = scaler.fit_transform(body_func_feature_sex_age[df_body_age_sub_columns])
body_func_feature_sex_age[df_body_age_sub_columns] = body_func_feature_sex_age_scaled
body_func_feature_sex_age['age'] = body_func_feature_sex_age['Date of Blood-0'] - body_func_feature_sex_age['Date of birth']

low, high = 40, 70
df_40_70 = body_func_feature_sex_age[(body_func_feature_sex_age['age'] > low) & (body_func_feature_sex_age['age'] <= high)]
print(low, high, len(df_40_70))
m = df_40_70['age'].mean()
s = df_40_70['age'].std()
df_40_70['label_age'] = (df_40_70['age'].values - m) / s

df_body_age_sub_columns = np.array(df_body_age_sub_columns)
np.save('instance0-1/df_body_age_sub_columns.npy', df_body_age_sub_columns)
df_body_age_sub_columns = np.load('instance0-1/df_body_age_sub_columns.npy')
joblib.dump(scaler, 'instance0-1/scaler.pkl')
age_mean_std = np.array([df_40_70['age'].mean(),df_40_70['age'].std()])
np.save('instance0-1/age_mean_std.npy', age_mean_std)

means = scaler.mean_
variances = scaler.var_
for i, col in enumerate(df_body_age_sub_columns):
    print(f"{col}: {means[i]}")
    print(f"{col}: {variances[i]}") 
# 计算每列的均值
temp = df_40_70[df_body_age_sub_columns].copy()
mean_values = temp.mean()

# 使用均值填充缺失值
temp.fillna(mean_values, inplace=True)
df_40_70_fillna = df_40_70.copy()
df_40_70_fillna[df_body_age_sub_columns] = temp
def elastic_grid_search(X, y, validation_folds=5, alpha_search=np.logspace(-4, -1, 4),
                        l1_ratio_search=np.linspace(0.01, 0.4, 3), n_bins=5, n_jobs=-1, random_state = 1716326954):

    binned_y = pd.cut(y, bins=n_bins, labels=False)
    kf = StratifiedKFold(n_splits=validation_folds)
    split = kf.split(binned_y, binned_y)

    param_grid = {'alpha': alpha_search,
                  'l1_ratio': l1_ratio_search,
                  }
    grid_model = GridSearchCV(ElasticNet(random_state=random_state, fit_intercept=True), param_grid, scoring="neg_mean_absolute_error",
                              cv=split, n_jobs=n_jobs)
    grid_model.fit(X, y) 
    print("Best l1_ratio " + str(grid_model.best_params_['l1_ratio']), flush=True)
    return grid_model

for key in list(organ_dict.keys())[1:]:
    # print(key)
    value = organ_dict[key]
    print(key,value)
    feat_organ = value + ['sex']
    filter_condition = df_40_70[feat_organ].count(axis=1)>= 3
    X_subset = df_40_70_fillna[filter_condition][feat_organ].values
    y_train_orginal = df_40_70_fillna[filter_condition]['age'].values
    y_train = df_40_70_fillna[filter_condition]['label_age'].values
    trained_model = elastic_grid_search(X_subset, y_train, validation_folds=5, n_jobs=-1, l1_ratio_search=[0.0, 0.2, 0.4, 0.8, 1.0])
    feature_results = {}
    feature_results[key] = {}
    feature_results[key]['features_coefficients'] = trained_model.best_estimator_.coef_
    feature_results[key]['intercept'] = trained_model.best_estimator_.intercept_
    feature_results[key]['best_alpha'] = trained_model.best_estimator_.alpha
    feature_results[key]['best_l1_ratio'] = trained_model.best_estimator_.l1_ratio
    feature_results[key]['feature'] = feat_organ

    new_elastic_net = ElasticNet(alpha=feature_results[key]['best_alpha'], l1_ratio=feature_results[key]['best_l1_ratio'])
    new_elastic_net.coef_ = feature_results[key]['features_coefficients']
    new_elastic_net.intercept_ = feature_results[key]['intercept']
    X_subset = pd.DataFrame(X_subset,columns = feat_organ)
    X_subset['age_pred'] = new_elastic_net.predict(X_subset)*s +m
    X_subset['age_diff'] = X_subset['age_pred'] - y_train_orginal
    b = np.polyfit(y_train_orginal, X_subset['age_diff'], 1)
    feature_results[key]['b'] = b
    X_subset['age_diff'] = X_subset['age_diff'] - np.polyval(b, y_train_orginal)
    X_subset['age'] = y_train_orginal

    fig = plt.figure(figsize=(5,5))
    reg_plot(fig.gca(), X_subset, f'age_pred', 'age', alpha=0.5, s=0.0001)
    plt.xlim(low, high)
    plt.savefig(f'instance0-1/figs/{key}.png')
    plt.close()
    with open(f'instance0-1/feature_results_{key}.pkl', 'wb') as f:
        pickle.dump(feature_results, f)
