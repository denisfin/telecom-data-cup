import numpy as np
import pandas as pd

import catboost as cat
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

from utils import preprocessing as prc


def plot_feature_importance(feature_importance, label='Features (avg over folds)'):
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                              by="importance", ascending=False)[:100].index
    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(14, 10))
    sns.barplot(x="importance", y="feature", data=best_features.groupby(['feature']).sum()
                .reset_index().sort_values(by="importance", ascending=False))
    plt.title(label)
    plt.tight_layout()
    plt.savefig('feature_importance.png')


def submit(test_df, preds):
    submission = test_df[['SK_ID']].copy()
    submission['predicted_CSI'] = preds
    grouped_test = submission.groupby('SK_ID', sort=False)['SK_ID', 'predicted_CSI'].mean()
    grouped_test['predicted_CSI'].to_csv('preds.csv', index=False)


def kfold_catboost(X, y, X_test, categorical_columns_index=None, params=None):
    # TRAIN MODELS
    model = cat.CatBoostClassifier(iterations=70,
                                   learning_rate=0.01,
                                   loss_function='Logloss',
                                   eval_metric='AUC',
                                   random_seed=params['my_random_state'])

    X_fold_train, X_fold_test, y_fold_train, y_fold_test = train_test_split(X, y, test_size=0.2, shuffle=False,
                                                                            random_state=params['my_random_state'])

    # KFold
    kfd = KFold(n_splits=5, shuffle=True, random_state=params['my_random_state'])
    preds = np.zeros(len(X_test))
    roc_auc = 0
    features = list(X.columns)
    feature_importance = pd.DataFrame()
    for num_fold, (train_index, val_index) in enumerate(kfd.split(X_fold_train, y_fold_train)):
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print("Training {} fold...".format(num_fold + 1))
        model.fit(X_train, y_train,
                  cat_features=categorical_columns_index,
                  eval_set=(X_val, y_val),
                  verbose=True,
                  metric_period=10,
                  use_best_model=True,
                  early_stopping_rounds=20)
        fpr, tpr, threshold = roc_curve(y_fold_test, model.predict_proba(X_fold_test)[:, 1])
        fold_auc = auc(fpr, tpr)
        print('ROC AUC Fold test: {}'.format(fold_auc))
        roc_auc += fold_auc / kfd.n_splits

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = features
        importance = pd.DataFrame({"importance": model.get_feature_importance()})
        fold_importance = pd.concat([fold_importance, importance], axis=1)
        fold_importance["fold"] = num_fold + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        feature_importance = feature_importance.fillna(0)

        preds += model.predict_proba(X_test)[:, 1] / kfd.n_splits

    print('-----------------')
    print('ROC AUC: {}'.format(roc_auc))

    # plot feature_importance
    plot_feature_importance(feature_importance, label='CatBoost Features (avg over folds)')

    return model, preds


def train(train_df, test_df, params):
    # to avoid overfitting
    train_df = train_df.sort_values('SK_ID')
    train_df = train_df.drop(('SK_ID'), axis=1)

    # separate X and Y
    (X, y) = prc.separate_xy(train_df, dropNaN=True)
    categorical_columns_index = [X.columns.get_loc(c) for c in train_df.columns if c in params['categorical_columns']]
    X_test = test_df.drop(['CSI', 'SK_ID'], axis=1)

    model, preds = kfold_catboost(X, y, X_test, categorical_columns_index, params)

    # submit
    submit(test_df, preds)

