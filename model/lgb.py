import numpy as np
import pandas as pd
import lightgbm as lgb

import time

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def score(ground_truth, predictions):
    return 1 / (1 + mean_squared_error(ground_truth, predictions) ** 0.5)


SCORE = make_scorer(score, greater_is_better=True)


def main(input_train='../data/TrainData.csv', input_pre='../data/PreData.csv'):
    TrainData = pd.read_csv(input_train)
    PreData = pd.read_csv(input_pre)
    TestId = PreData['Id']
    params = dict(learning_rate=0.05, boosting_type='gbdt', objective='regression', metric='mse', sub_feature=0.7,
                  num_leaves=40, colsample_bytree=0.7, feature_fraction=0.7, min_data=100, min_hessian=1, verbose=-1)

    test_preds = np.zeros((PreData.shape[0], 5))
    cv_pre = np.zeros((20000, 5))
    print('开始CV 5折训练...')
    kf = KFold(n_splits=5, shuffle=True, random_state=2018)
    for i, (train_index, test_index) in enumerate(kf.split(TrainData)):
        print('第{}次训练...'.format(i + 1))
        train_feat1 = TrainData.iloc[train_index]
        train_feat2 = TrainData.iloc[test_index]
        lgb_train1 = lgb.Dataset(train_feat1.drop(['Score', 'Id'], axis=1), train_feat1['Score'])
        lgb_train2 = lgb.Dataset(train_feat2.drop(['Score', 'Id'], axis=1), train_feat2['Score'])
        gbm = lgb.train(params,
                        lgb_train1,
                        num_boost_round=3000,
                        valid_sets=lgb_train2,
                        verbose_eval=100,
                        early_stopping_rounds=50)
        y_pred = gbm.predict(train_feat2.drop(['Score', 'Id'], axis=1))
        cv_pre[:, i] = y_pred
        print('Score: %.4f' % (1 / (1 + np.sqrt(mean_squared_error(train_feat2['Score'], y_pred.round())))))
        test_preds[:, i] = gbm.predict(PreData.drop(['Id'], axis=1))

    submission = pd.DataFrame({'Id': TestId, 'Score': test_preds.mean(axis=1)})
    submission['Score'] = submission['Score'].round()
    submission['Score'] = submission['Score'].astype('int64')
    submission.to_csv('E:/AIcompetition/DataFountainScore/result/Lgb.csv', header=None, index=False)


def model_tuning(input_train='../data/TrainData.csv'):
    start_time = time.time()

    TrainData = pd.read_csv(input_train)

    print('开始调优...')
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.08],
        'num_leaves': [30, 40, 50],
        'n_estimators': [300, 500],
        'max_depth': [7, 9, 11]
    }

    clf = lgb.LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=40,
        max_depth=7,
        learning_rate=0.05,
        n_estimators=500,
        objective='regression',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2018,
    )
    model_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=2, cv=5, scoring=SCORE, verbose=10)
    model_search.fit(TrainData.drop(['Score', 'Id'], axis=1), TrainData['Score'])
    print('--- Grid Search Completed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Param grid:')
    print(param_grid)
    print('Best Params:')
    print(model_search.best_params_)
    print('Best CV Score:')
    print(model_search.best_score_)


if __name__ == '__main__':
    main()
    # model_tuning()
