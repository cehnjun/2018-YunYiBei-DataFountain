import pandas as pd
import numpy as np
import time

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


def score(ground_truth, predictions):
    return 1 / (1 + mean_squared_error(ground_truth, predictions) ** 0.5)


SCORE = make_scorer(score, greater_is_better=True)


def main(input_train='../data/TrainData.csv', input_pre='../data/PreData.csv'):
    TrainData = pd.read_csv(input_train)
    PreData = pd.read_csv(input_pre)
    TestId = PreData['Id']
    test_preds = np.zeros((PreData.shape[0], 5))
    cv_pre = np.zeros((20000, 5))
    model_ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=True, tol=0.001)
    print('开始CV 5折训练...')
    kf = KFold(n_splits=5, shuffle=True, random_state=2018)
    for i, (train_index, test_index) in enumerate(kf.split(TrainData)):
        print('第{}次训练...'.format(i + 1))
        train_feat1 = TrainData.iloc[train_index]
        train_feat2 = TrainData.iloc[test_index]
        model_ridge.fit(train_feat1.drop(['Score', 'Id'], axis=1), train_feat1['Score'])
        y_pred = model_ridge.predict(train_feat2.drop(['Score', 'Id'], axis=1))
        cv_pre[:, i] = y_pred
        print('Score: %.4f' % (1 / (1 + np.sqrt(mean_squared_error(train_feat2['Score'], y_pred.round())))))
        test_preds[:, i] = model_ridge.predict(PreData)

    submission = pd.DataFrame({'Id': TestId, 'Score': test_preds.mean(axis=1)})
    submission['Score'] = submission['Score'].round()
    submission['Score'] = submission['Score'].astype('int64')
    submission.to_csv('E:/AIcompetition/DataFountainScore/data/ridge.csv', header=None, index=False)


def model_tuning(input_train='../data/TrainData.csv'):
    start_time = time.time()

    TrainData = pd.read_csv(input_train)

    print('开始调优...')
    param_grid = {
        'alpha': [0.4, 0.5, 0.6],
    }

    clf = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=True, tol=0.001)
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
