import time

start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def score(ground_truth, predictions):
    return 1 / (1 + mean_squared_error(ground_truth, predictions) ** 0.5)


SCORE = make_scorer(score, greater_is_better=True)


class Ensemble:
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=2018)
        s_train = np.zeros((x.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            print('Fitting For Base Model #%d / %d ---', i + 1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds.split(x)):
                print('--- Fitting For Fold %d / %d ---', j + 1, self.n_splits)
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_holdout = x[test_idx]
                clf.fit(x_train, y_train)
                y_prediction = clf.predict(x_holdout)[:]
                s_train[test_idx, i] = y_prediction

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        clf = self.stacker
        clf.fit(s_train, y)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def predict(self, x):
        x = np.array(x)
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=2018)
        s_test = np.zeros((x.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            s_test_i = np.zeros((x.shape[0], folds.n_splits))
            for j, (train_idx, test_idx) in enumerate(folds.split(x)):
                s_test_i[:, j] = clf.predict(x)[:]
            s_test[:, i] = s_test_i.mean(1)

        clf = self.stacker
        y_prediction = clf.predict(s_test)[:]
        return y_prediction

    def fit_predict(self, x, y, t):
        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=2018)

        s_train = np.zeros((x.shape[0], len(self.base_models)))
        s_test = np.zeros((t.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #{0} / {1} ---'.format(i + 1, len(self.base_models)))

            s_test_i = np.zeros((t.shape[0], folds.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds.split(x)):
                print('--- Fitting For Fold #{0} / {1} ---'.format(j + 1, self.n_splits))

                x_train = x[train_idx]
                y_train = y[train_idx]
                x_holdout = x[test_idx]
                clf.fit(x_train, y_train)
                y_prediction = clf.predict(x_holdout)[:]
                s_train[test_idx, i] = y_prediction
                s_test_i[:, j] = clf.predict(t)[:]

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            s_test[:, i] = s_test_i.mean(1)

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        # param_grid = {
        #     'n_estimators': [100],
        #     'learning_rate': [0.45, 0.05, 0.055],
        #     'subsample': [0.72, 0.75, 0.78]
        # }
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05],
            'subsample': [0.75]
        }
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=SCORE)
        grid.fit(s_train, y)

        # a little memo
        message = 'to determine local CV score of #28'

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_prediction = grid.predict(s_test)[:]

        return y_prediction
