import csv
import os
import warnings
from enum import Enum

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import feature_meta
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedShuffleSplit
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# In my version, the following isn't ignored by default.
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


REBUILD_FEATURES = False
RE_PICK_FTR = False

CHOSEN_FEATURES = feature_meta.NODE_FEATURES
# CHOSEN_FEATURES = {"multi_dimensional_scaling": FeatureMeta(MultiDimensionalScaling, {"mds"})}
PATH = os.path.join("..", "data_by_community")


class LearningMethod(Enum):
    RF = "random_forest"
    SVM = "support_vector_machine"
    XGBOOST = "XG_Boost"


class MLCommunities:
    def __init__(self, method=LearningMethod.XGBOOST):
        self.labels = None
        self._beta_pairs = None
        self._beta_matrix = None
        self._nodes = None
        self._edges = None
        self._best_beta_df = None
        self._method = method

    def forward_time_data(self, beta_matrix, nodes, edges, labels):
        self.labels = labels
        self._beta_pairs = [i for i in range(beta_matrix.shape[1])]
        self._beta_matrix = beta_matrix
        self._nodes = nodes
        self._edges = edges
        # self._best_beta_df = self._best_pairs_df()
        self._best_beta_df = self._beta_matrix_to_df(self._beta_pairs)

    def run(self):
        if self._method.value == LearningMethod.RF.value:
            self._learn_RF(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=10))
        if self._method.value == LearningMethod.SVM.value:
            self._learn_SVM(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=5))
        if self._method.value == LearningMethod.XGBOOST.value:
            return self._learn_XGBoost_p(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=10))

    def _beta_matrix_to_df(self, header):
        # create header
        return pd.DataFrame(data=self._beta_matrix, columns=header)

    def _pca_df(self, beta_df, n_components=20, graph_data=False, min_nodes=None):
        pca = PCA(n_components=n_components)

        if min_nodes:
            beta_df_temp = beta_df.copy()
            beta_df_temp['nodes'] = self._nodes
            beta_df_temp['edges'] = self._edges
            beta_df_temp['labels'] = self.labels
            beta_df_temp = beta_df_temp[beta_df_temp.nodes >= min_nodes]
            self.labels = beta_df_temp['labels'].tolist()
            self._nodes = beta_df_temp['nodes'].tolist()
            self._edges = beta_df_temp['edges'].tolist()
            beta_df_temp = beta_df_temp.drop(['nodes', 'labels'], axis=1)
            beta_df = beta_df_temp

        if graph_data:
            # add edge and node number
            # not taking pca, only removing too small graphs.
            # not pca.fit_transform(beta_df).
            return np.hstack([beta_df, np.matrix(self._nodes).T, np.matrix(self._edges).T])

        return beta_df

    def _learn_XGBoost_p(self, principalComponents):
        if not os.path.exists(os.path.join(os.getcwd(), 'parameter_check')):
            os.mkdir('parameter_check')
        # train percentage
        for train_p in [70]:
            f = open(os.path.join(os.getcwd(), 'parameter_check', "results_train_p"+str(train_p)+"dart"+".csv"), 'w')
            w = csv.writer(f)
            w.writerow(['max_depth', 'lambda', 'eta', 'min child weight', 'subsample', 'ntree_limit', 'sample type',
                        'normalize type', 'rate drop', 'train_AUC', 'test_AUC'])
            for max_depth, l, eta, min_child_weight, subsample, ntree_limit, sample_type, normalize_type, rate_drop in \
                    itertools.product(range(3, 12, 4), range(1, 26, 6), np.logspace(-3, -0.5, 5), range(5, 21, 5),
                                      range(5, 11), range(1, 121, 40), ['uniform', 'weighted'], ['tree', 'forest'],
                                      range(2, 9, 2)):
                auc_train = []
                auc_test = []
                for num_splits in range(1, 101):
                    X_train, X_test, y_train, y_test = train_test_split(principalComponents, self.labels,
                                                                        test_size=1-float(train_p)/100)
                    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1)
                    dtrain = xgb.DMatrix(X_train, y_train, silent=True)
                    dtest = xgb.DMatrix(X_test, y_test, silent=True)
                    deval = xgb.DMatrix(X_eval, y_eval, silent=True)
                    params = {'silent': True, 'booster': 'dart', 'tree_method': 'gpu_hist', 'max_depth': max_depth,
                              'lambda': l/10, 'eta': eta, 'min_child_weight': min_child_weight,
                              'subsample': subsample/10, 'sample_type': sample_type, 'normalize_type': normalize_type,
                              'rate_drop': rate_drop/10}
                    clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                        early_stopping_rounds=10, verbose_eval=False)
                    y_score_test = clf_xgb.predict(dtest, ntree_limit=ntree_limit)
                    y_score_train = clf_xgb.predict(dtrain, ntree_limit=ntree_limit)
                    # ROC AUC has a problem with only one class
                    try:
                        r1 = roc_auc_score(y_test, y_score_test)
                    except ValueError:
                        continue
                    auc_test.append(r1)

                    try:
                        r2 = roc_auc_score(y_train, y_score_train)
                    except ValueError:
                        continue
                    auc_train.append(r2)
                w.writerow([str(max_depth), str(l/10), str(eta), str(min_child_weight),
                            str(subsample/10), str(ntree_limit), str(sample_type), str(normalize_type),
                            str(rate_drop/10), str(np.mean(auc_train)), str(np.mean(auc_test))])
        return None

    def _learn_XGBoost(self, principalComponents):
        df = pd.DataFrame()
        # train percentage
        for train_p in range(30, 90, 10):
            auc_train = []
            auc_test = []
            for num_splits in range(1, 501):
                X_train, X_test, y_train, y_test = train_test_split(principalComponents, self.labels,
                                                                    test_size=1-float(train_p)/100)
                X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.1)
                dtrain = xgb.DMatrix(X_train, y_train, silent=True)
                dtest = xgb.DMatrix(X_test, y_test, silent=True)
                deval = xgb.DMatrix(X_eval, y_eval, silent=True)
                params = {'silent': True, 'booster': 'gblinear', 'lambda': 0.7, 'eta': 0.39}
                clf_xgb = xgb.train(params, dtrain=dtrain, evals=[(dtrain, 'train'), (deval, 'eval')],
                                    early_stopping_rounds=10, verbose_eval=False)
                y_score_test = clf_xgb.predict(dtest)
                y_score_train = clf_xgb.predict(dtrain)
                # ROC AUC has a problem with only one class
                try:
                    r1 = roc_auc_score(y_test, y_score_test)
                except ValueError:
                    continue
                auc_test.append(r1)

                try:
                    r2 = roc_auc_score(y_train, y_score_train)
                except ValueError:
                    continue
                auc_train.append(r2)
            df1 = pd.DataFrame([[train_p, np.mean(auc_train), np.mean(auc_test)]],
                               columns=['train_p', 'train_auc', 'test_auc'])
            df = pd.concat([df, df1])
            print(['train: ' + str(train_p) +'%', 'Train AUC: ' + str(np.mean(auc_train)),
                   'Test AUC: ' + str(np.mean(auc_test))])
        return df

    def _learn_SVM(self, principalComponents):
        df = pd.DataFrame(columns=['C', 'train_p', 'mean_auc'])
        # penalty for svm
        for C in np.logspace(-2, 2, 5):
            # train percentage
            for train_p in range(5, 90, 10):
                cv = ShuffleSplit(n_splits=1, test_size=1 - float(train_p) / 100)
                clf_svm = SVC(C=C, kernel='linear', probability=False, shrinking=False,
                              class_weight='balanced')
                # print(clf_svm)
                # clf_RF = RandomForestClassifier()
                scores_svm = cross_val_score(clf_svm, principalComponents, self.labels, cv=cv, scoring='roc_auc')
                # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                df.loc[len(df)] = [C, train_p, np.mean(scores_svm)]
                print([C, train_p, np.mean(scores_svm)])
        return df

    def _learn_RF(self, principalComponents):
        df = pd.DataFrame(columns=['train_p', 'mean_auc'])
        # train percentage
        for train_p in range(30, 90, 10):
            cv = StratifiedShuffleSplit(n_splits=1, test_size=1 - float(train_p) / 100)
            clf_rf = RandomForestClassifier(n_estimators=200, max_features="log2", criterion="gini", max_depth=15)
            # print(clf_svm)
            # clf_RF = RandomForestClassifier()
            scores_rf = cross_val_score(clf_rf, principalComponents, self.labels, cv=cv, scoring='roc_auc')
            # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
            df.loc[len(df)] = [train_p, np.mean(scores_rf)]
            print([train_p, np.mean(scores_rf)])
        return df

    def plot_learning_df(self, df):
        new_df = pd.DataFrame(df[df['rf-max_depth'] == 9], )
        new_df.reset_index()
        new_df.plot(x='train_p', y='mean_auc')
        plt.savefig("auc.jpg")
        plt.show()
