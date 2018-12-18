import numpy as np
import pandas as pd
import os
from collections import Counter
from datetime import datetime as dt

import catboost as cat
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# DATA
# load row data
my_random_state = 255
n_rows = None  # 'None' for all read
base_folder = 'D:\PythonProjects/condition_csi/dataset'
train_folder = os.path.join(base_folder, 'train')
test_folder = os.path.join(base_folder, 'test')
load_preproc_csv = True

if load_preproc_csv == False:
    # train
    csi_train = pd.read_csv(os.path.join(train_folder, 'subs_csi_train.csv'), sep=';', nrows=n_rows)
    features_train = pd.read_csv(os.path.join(train_folder, 'subs_features_train.csv'), sep=';', nrows=n_rows)

    consum_train = pd.read_csv(os.path.join(train_folder, 'subs_bs_consumption_train.csv'), sep=';', nrows=n_rows)
    voice_session_train = pd.read_csv(os.path.join(train_folder, 'subs_bs_voice_session_train.csv'), sep=';',
                                      nrows=n_rows)
    data_session_train = pd.read_csv(os.path.join(train_folder, 'subs_bs_data_session_train.csv'), sep=';',
                                     nrows=n_rows)

    # merged different csv
    # data_train.loc[data_train['SK_ID'] == 3851]
    data_train = pd.merge(csi_train, features_train, left_on='SK_ID', right_on='SK_ID', how='left')
    # del csi_train, features_train

    SNAP_DATE_formated = data_train['SNAP_DATE'].apply(lambda x: dt.strptime(str(x), '%d.%m.%y'))
    data_train.loc[:, 'SNAP_DATE_month'] = SNAP_DATE_formated.apply(lambda x: x.month)
    del data_train['SNAP_DATE']
    consum_train['MON'] = consum_train['MON'].apply(lambda x: dt.strptime(str(x), '%d.%m').month)
    data_train = pd.merge(data_train, consum_train, left_on=['SK_ID', 'SNAP_DATE_month'],
                          right_on=['SK_ID', 'MON'], how='left')  # ПОПРОБОВАТЬ how='left', если мало будет данных
    del data_train['MON']
    # del consum_train

    START_TIME_formated = voice_session_train['START_TIME'].apply(lambda x: dt.strptime(str(x), '%d.%m %H:%M:%S'))
    voice_session_train.loc[:, 'voice_session_START_TIME_day'] = START_TIME_formated.apply(lambda x: x.day).astype(int)
    voice_session_train.loc[:, 'voice_session_START_TIME_month'] = START_TIME_formated.apply(lambda x: x.month)
    del voice_session_train['START_TIME']
    voice_session_train['VOICE_DUR_MIN'] = ('0' + voice_session_train['VOICE_DUR_MIN']
                                            .str.replace(',', '.')).astype(float)
    voice_session_train = voice_session_train.groupby(['voice_session_START_TIME_day',
                                                       'voice_session_START_TIME_month', 'SK_ID',
                                                       'CELL_LAC_ID'], sort=False).sum().reset_index()
    voice_session_train = voice_session_train[voice_session_train['VOICE_DUR_MIN'] > 0]

    data_train = pd.merge(data_train, voice_session_train, left_on=['SK_ID', 'CELL_LAC_ID', 'SNAP_DATE_month'],
                          right_on=['SK_ID', 'CELL_LAC_ID', 'voice_session_START_TIME_month'], how='left')
    del data_train['voice_session_START_TIME_month']
    # del voice_session_train

    START_TIME_formated = data_session_train['START_TIME'].apply(lambda x: dt.strptime(str(x), '%d.%m %H:%M:%S'))
    data_session_train.loc[:, 'data_session_START_TIME_day'] = START_TIME_formated.apply(lambda x: x.day)
    data_session_train.loc[:, 'data_session_START_TIME_month'] = START_TIME_formated.apply(lambda x: x.month)
    del data_session_train['START_TIME']
    data_session_train['DATA_VOL_MB'] = ('0' + data_session_train['DATA_VOL_MB'].str.replace(',', '.')).astype(float)

    data_session_train = data_session_train.groupby(['data_session_START_TIME_day',
                                                     'data_session_START_TIME_month', 'SK_ID',
                                                     'CELL_LAC_ID'], sort=False).sum().reset_index()
    data_session_train = data_session_train[data_session_train['DATA_VOL_MB'] > 0]

    data_train = pd.merge(data_train, data_session_train, left_on=['SK_ID', 'CELL_LAC_ID',
                          'voice_session_START_TIME_day', 'SNAP_DATE_month'], right_on=['SK_ID', 'CELL_LAC_ID',
                          'data_session_START_TIME_day', 'data_session_START_TIME_month'], how='left')
    del data_train['data_session_START_TIME_day']
    del data_train['data_session_START_TIME_month']
    # del data_session_train

    # test
    csi_test = pd.read_csv(os.path.join(test_folder, 'subs_csi_test.csv'), sep=';', nrows=n_rows)
    features_test = pd.read_csv(os.path.join(test_folder, 'subs_features_test.csv'), sep=';', nrows=n_rows)

    consum_test = pd.read_csv(os.path.join(test_folder, 'subs_bs_consumption_test.csv'), sep=';', nrows=n_rows)
    voice_session_test = pd.read_csv(os.path.join(test_folder, 'subs_bs_voice_session_test.csv'), sep=';', nrows=n_rows)
    data_session_test = pd.read_csv(os.path.join(test_folder, 'subs_bs_data_session_test.csv'), sep=';', nrows=n_rows)

    # merged different csv
    data_test = pd.merge(csi_test, features_test, left_on='SK_ID', right_on='SK_ID', how='left')
    # del csi_test, features_test

    SNAP_DATE_formated = data_test['SNAP_DATE'].apply(lambda x: dt.strptime(str(x), '%d.%m.%y'))
    data_test.loc[:, 'SNAP_DATE_month'] = SNAP_DATE_formated.apply(lambda x: x.month)
    del data_test['SNAP_DATE']
    consum_test['MON'] = consum_test['MON'].apply(lambda x: dt.strptime(str(x), '%d.%m').month)
    data_test = pd.merge(data_test, consum_test, left_on=['SK_ID', 'SNAP_DATE_month'],
                         right_on=['SK_ID', 'MON'], how='left')
    del data_test['MON']
    # del consum_test

    START_TIME_formated = voice_session_test['START_TIME'].apply(lambda x: dt.strptime(str(x), '%d.%m %H:%M:%S'))
    voice_session_test.loc[:, 'voice_session_START_TIME_day'] = START_TIME_formated.apply(lambda x: x.day)
    voice_session_test.loc[:, 'voice_session_START_TIME_month'] = START_TIME_formated.apply(lambda x: x.month)
    del voice_session_test['START_TIME']
    voice_session_test['VOICE_DUR_MIN'] = ('0' + voice_session_test['VOICE_DUR_MIN']
                                           .str.replace(',', '.')).astype(float)
    voice_session_test = voice_session_test.groupby(['voice_session_START_TIME_day',
                                                     'voice_session_START_TIME_month', 'SK_ID',
                                                     'CELL_LAC_ID'], sort=False).sum().reset_index()
    voice_session_test = voice_session_test[voice_session_test['VOICE_DUR_MIN'] > 0]

    data_test = pd.merge(data_test, voice_session_test, left_on=['SK_ID', 'CELL_LAC_ID', 'SNAP_DATE_month'],
                         right_on=['SK_ID', 'CELL_LAC_ID', 'voice_session_START_TIME_month'], how='left')
    del data_test['voice_session_START_TIME_month']
    # del voice_session_test

    START_TIME_formated = data_session_test['START_TIME'].apply(lambda x: dt.strptime(str(x), '%d.%m %H:%M:%S'))
    data_session_test.loc[:, 'data_session_START_TIME_day'] = START_TIME_formated.apply(lambda x: x.day)
    data_session_test.loc[:, 'data_session_START_TIME_month'] = START_TIME_formated.apply(lambda x: x.month)
    del data_session_test['START_TIME']
    data_session_test['DATA_VOL_MB'] = ('0' + data_session_test['DATA_VOL_MB'].str.replace(',', '.')).astype(float)
    data_session_test = data_session_test.groupby(['data_session_START_TIME_day',
                                                   'data_session_START_TIME_month', 'SK_ID',
                                                   'CELL_LAC_ID'], sort=False).sum().reset_index()
    data_session_test = data_session_test[data_session_test['DATA_VOL_MB'] > 0]

    data_test = pd.merge(data_test, data_session_test, left_on=['SK_ID', 'CELL_LAC_ID',
                         'voice_session_START_TIME_day', 'SNAP_DATE_month'], right_on=['SK_ID', 'CELL_LAC_ID',
                         'data_session_START_TIME_day', 'data_session_START_TIME_month'], how='left')
    del data_test['data_session_START_TIME_day']
    del data_test['data_session_START_TIME_month']
    # del data_session_test

    # common
    # bs_avg_kpi = pd.read_csv(os.path.join(base_folder, 'bs_avg_kpi.csv'), sep=';', nrows=n_rows)
    # bs_chnn_kpi = pd.read_csv(os.path.join(base_folder, 'bs_chnn_kpi.csv'), sep=';', nrows=n_rows)

    data_train.to_csv('data/data_train.csv', index=False)
    data_test.to_csv('data/data_test.csv', index=False)
else:
    data_train = pd.read_csv('data/data_train.csv')
    data_test = pd.read_csv('data/data_test.csv')
    # merge train and test data
    # Preprocessing Merged Train and Test data
    data = pd.concat([data_train, data_test])
    trn_len = data_train.shape[0]
    del data_train, data_test

    # data_sorted = data.sort_values(by=['SK_ID'])
    print(data.shape)
    print(data.info())

    # data preprocessing
    categorical_columns = ['COM_CAT#1', 'COM_CAT#2', 'COM_CAT#3', 'BASE_TYPE', 'ACT', 'ARPU_GROUP', 'COM_CAT#7',
                           'DEVICE_TYPE_ID', 'INTERNET_TYPE_ID', 'COM_CAT#25', 'COM_CAT#26', 'COM_CAT#34']
    date_columns = ['CONTACT_DATE', 'voice_session_START_TIME_day']
    date_columns_formated = ['SNAP_DATE', 'MON', 'START_TIME', 'SNAP_DATE_month']
    numerical_columns_float = ['REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST', 'COM_CAT#8', 'COM_CAT#17',
                               'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21', 'COM_CAT#22', 'COM_CAT#23',
                               'COM_CAT#24', 'COM_CAT#27', 'COM_CAT#28', 'COM_CAT#29', 'COM_CAT#30', 'COM_CAT#31',
                               'COM_CAT#32', 'COM_CAT#33',
                               'SUM_MINUTES', 'SUM_DATA_MB', 'SUM_DATA_MIN', 'VOICE_DUR_MIN', 'DATA_VOL_MB']
    numerical_columns_int = ['CELL_LAC_ID']
    # delete_columns = ['COM_CAT#3', 'COM_CAT#24']

    # delete  non-informative columns
    # for del_col in delete_columns:
    #     del data[del_col]

    # categorical columns
    for cat_col in categorical_columns:
        if (data[cat_col].count() != data.shape[0]):
            data[cat_col] = data[cat_col].fillna(data[cat_col].median(axis=0), axis=0)

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[cat_col].values.astype('str')))
        data.loc[:, cat_col] = lbl.transform(list(data[cat_col].values.astype('str')))

    # numerical columns float (fillna, mean, std)
    for num_col in numerical_columns_float:
        if data[num_col].dtypes == 'object':
            data[num_col] = ('0' + data[num_col].str.replace(',', '.')).astype(float)
        if (data[num_col].count() != data.shape[0]):
            data[num_col] = data[num_col].fillna(0)
        # data[num_col] = (data[num_col] - data[num_col].mean()) / data[num_col].std()

    # numerical columns int (fillna, mean, std)
    for num_col in numerical_columns_int:
        if (data[num_col].count() != data.shape[0]):
            data[num_col] = data[num_col].fillna(0)
        data[num_col] = data[num_col].astype(int)

    # date columns
    for date_col in date_columns:
        counter = Counter(data[date_col].tolist())
        average_value = counter.most_common(1)[0][0]
        data[date_col] = data[date_col].fillna(average_value, axis=0)

        if date_col == 'voice_session_START_TIME_day':
            data[date_col] = data[date_col].fillna(1, axis=0)
            data[date_col] = data[date_col].astype(int)
            continue

        if date_col == 'CONTACT_DATE':
            format_str = '%d.%m'
        elif date_col == 'SNAP_DATE':
            format_str = '%d.%m.%y'
        else:
            format_str = '%Y%m%d'
        date_col_formated = data[date_col].apply(lambda x: dt.strptime(str(x), format_str))
        data.loc[:, date_col + '_month'] = date_col_formated.apply(lambda x: x.month)
        data.loc[:, date_col + '_day'] = date_col_formated.apply(lambda x: x.day)
        # data.loc[:, date_col + '_weekday'] = date_col_formated.apply(lambda x: x.weekday())
        del date_col_formated
        del data[date_col]

    # delete non-informative columns count_unique = 1
    for c in data.columns:
        num_unique = len(data[c].unique())
        print("Count unique '{0}': {1}".format(c, num_unique))
        if num_unique < 2:
            data = data.drop(columns=[c])

    # Strength features
    # data['COM_CAT#20+COM_CAT#22'] = 0.5*data['COM_CAT#20'] + 0.5*data['COM_CAT#22']
    # data['REVENUE+COM_CAT#23'] = 0.5*data['REVENUE'] + 0.5*data['COM_CAT#23']
    # data['ITC+COM_CAT#28'] = 0.5*data['ITC'] + 0.5*data['COM_CAT#28']
    # data = data.drop(columns=['ITC', 'REVENUE', 'COM_CAT#20', 'COM_CAT#22', 'COM_CAT#23'], axis=1)

    print(data.info())
    # scatter_matrix(data, alpha=0.05)
    corr = data.corr()

    # Back split data
    train_df = data[:trn_len]
    test_df = data[trn_len:]
    train_df.to_csv('data/train_df.csv', index=False)
    test_df.to_csv('data/test_df.csv', index=False)


# separate X and Y
# Удалить SK_ID для обучения
X = train_df.drop(('CSI'), axis=1)
train_df['CSI'] = train_df['CSI'].fillna(0)  # ПОПРОБОВАТЬ УДАЛИТЬ Nan
y = train_df['CSI'].astype(int)
print('Labels: {}'.format(set(y)))
print('Zero count: {}, One count: {}'.format(len(y) - sum(y), sum(y)))
# Необходимо убирать дисбаланс в классах
X_test = test_df.drop(('CSI'), axis=1)

# TRAIN MODELS
model = cat.CatBoostClassifier(iterations=100,
                               # use_best_model=True,
                               # early_stopping_rounds=20,
                               learning_rate=0.01,
                               loss_function='Logloss',
                               eval_metric='AUC',
                               random_seed=my_random_state)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=my_random_state)
# pool_train = cat.Pool(data=X_train, label=y_train, cat_features=categorical_columns)
# pool_val = cat.Pool(data=X_val, label=y_val, cat_features=categorical_columns)
categorical_columns_index = [X.columns.get_loc(c) for c in data.columns if c in categorical_columns]

# GridSearchCV
grid_search = False

X_fold_train, X_fold_test, y_fold_train, y_fold_test = train_test_split(X, y, test_size=0.2,
                                                                        random_state=my_random_state)
# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=my_random_state)
preds = np.zeros(len(X_test))
auc = 0
features = list(X.columns)
feature_importance = pd.DataFrame()
for num_fold, (train_index, val_index) in enumerate(skf.split(X_fold_train, y_fold_train)):
    print("Trianing {} fold...".format(num_fold + 1))
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    if grid_search == True:
        # GridSearchCV
        params_grid = {'iterations': np.arange(100, 1000, 10),
                       'depth': np.arange(4, 12, 2),
                       'learning_rate': [0.01, 0.03, 0.1],
                       'l2_leaf_reg': [3, 1, 5, 10, 100],

                       'loss_function': 'Logloss',
                       'random_seed': [0, 128, 255]}

        grid = GridSearchCV(model, param_grid=params_grid)
        grid.fit(X_train, y_train)
        print('CV error             = ', 1 - grid.best_score_)
        print('best iterations      = ', grid.best_estimator_.iterations)
        print('best depth           = ', grid.best_estimator_.depth)
        print('best learning_rate   = ', grid.best_estimator_.learning_rate)
    else:
        model.fit(X_train, y_train,
                  cat_features=categorical_columns_index,
                  eval_set=(X_val, y_val),
                  verbose=True,
                  metric_period=10)
        fold_auc = roc_auc_score(y_fold_test, model.predict_proba(X_fold_test)[:, 1])
        print('ROC AUC Fold test: {}'.format(fold_auc))
        auc += fold_auc / skf.n_splits

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features
        importance = pd.DataFrame({"importance": model.get_feature_importance()})
        fold_importance = pd.concat([fold_importance, importance], axis=1)
        fold_importance["fold"] = num_fold + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    feature_importance = feature_importance.fillna(0)
    preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

print('ROC AUC: {}'.format(auc))
if grid_search == True:
    exit()

# plot feature_importance
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:100].index
best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(14, 10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('CAT Features (avg over folds)')
plt.tight_layout()
plt.savefig('feature_importance.png')

# submit
submission = test_df[['SK_ID']].copy()
submission['predicted_CSI'] = preds
grouped_test = submission.groupby('SK_ID', sort=False)['SK_ID', 'predicted_CSI'].mean()
grouped_test['predicted_CSI'].to_csv('preds.csv', index=False)
print('end')

# Не фиксировать seed, а пробовать много разных
# Генетический алгоритм для отбора признаков
# StratifiedKFold если классы не сбалансированы
# Стекинг моделей
# GridSearchCV или RandomizedSearchCV для настройки параметров
# Feature Evaluation
# проверить распределние y
# посмотреть на каких метках ошибается
# Проверить обычный KFold
# GridSearchCV
# Параметр catboost one hot max size и l2_leaf_reg

# Features
# test features
# поменять местами merge таблиц
# расчитать средние показатели VOICE_DUR_MIN для каждого SK_ID, усреднение по месяцам/дням
