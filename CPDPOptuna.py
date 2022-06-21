import numpy as np
from optuna import Study
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from cpdp import CPDP
import optuna


class CPDPOptuna:
    SVM_kernel = 'poly'
    SVM_degree = 3
    SVM_gamma = 1
    SVM_c = 1
    SVM_coef0 = 0

    RF_criterion = 'gini'
    RF_n_estimators = 10
    RF_max_features = 'auto'
    RF_min_samples_split = 2

    boost_learning_rate = 1
    boost_n_estimators = 50

    KNN_n_neighbors = 5

    def __init__(self, cpdp_method: str, classifier: str, source_x: np.array, source_y: np.array, target_x: np.array,
                 target_y: np.array, num_of_times: int = 10, n_trials: int = 1000):
        self.cpdp_method = cpdp_method
        self.classifier = classifier
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_y = target_y
        self.num_of_times = num_of_times
        self.n_trials = n_trials

    def _predict_target(self, source_x: np.array, source_y: np.array, target_x: np.array):
        """
        This method aim to build a model based on a classifier specified in self.classifier and use target_x for
        prediction.
        :param source_x: Numpy array that has source data projects for X
        :param source_y: Numpy array that has source data projects for Y (target feature)
        :param target_x: Numpy array that has target data project for X
        :return: model.predict result for target_x
        """

        if self.classifier == 'KNN':
            model = KNeighborsClassifier(n_neighbors=self.KNN_n_neighbors)
        elif self.classifier == 'Boost':
            model = AdaBoostClassifier(n_estimators=self.boost_n_estimators, learning_rate=self.boost_learning_rate)
        elif self.classifier == 'SVM':
            model = SVR(kernel=self.SVM_kernel, degree=self.SVM_degree, gamma=self.SVM_gamma, C=self.SVM_c,
                        coef0=self.SVM_coef0)
        elif self.classifier == 'RF':
            model = RandomForestClassifier(n_estimators=self.RF_n_estimators, criterion=self.RF_criterion,
                                           max_features=self.RF_max_features,
                                           min_samples_split=self.RF_min_samples_split)
        else:
            raise "Unknown Classifier"

        model.fit(source_x, source_y)
        if self.classifier == 'KNN':
            if source_x.shape[0] < self.KNN_n_neighbors:
                return 0
        return model.predict(target_x)

    def objective(self, trial):
        n_neighbors = trial.suggest_int("k_n_neighbors", 2, 101, log=True)
        if self.cpdp_method == 'Bruakfilter':
            cpdp_obj = CPDP.Bruakfilter(n_neighbors=n_neighbors)
        elif self.cpdp_method == 'DSBF':
            topk = trial.suggest_int("topk", 1, 10, log=True)
            cpdp_obj = CPDP.DSBF(topK=topk, neighbors=n_neighbors)
        else:
            raise "Unknown CPDP method"

        source_x, source_y, target_x, target_y = cpdp_obj.run(self.source_x, self.source_y, self.target_x,
                                                              self.target_y)
        target_x.astype(float)
        target_y.astype(float)
        predict = self._predict_target(source_x, source_y, target_x)
        return roc_auc_score(target_y, predict)
        #     score = cross_val_score(knn, X_train, y_train, n_jobs=-1, cv=3)
        #     accuracy = score.mean()
        #     return accuracy
        #     # accuracy = score.mean()
        #     # return accuracy

    def _export_to_file(self, study: Study):
        """
        This method is implemented to write results for best parameter to a txt file.
        :param study: a Study object that will be used to get the best trial
        :return: None
        """
        trial = study.best_trial
        with open(f'{self.cpdp_method}_{self.classifier}.txt', 'a') as file:
            file.write(f'AUC: {trial.value}')
            file.write(f'Best hyperparameters: {trial.params}')
            file.write(f'Duration: {trial.duration}')

    def run(self):
        """
        This method is to run Optuna to find the optimal parameters for specific CPDP method and Classifier.
        :return: None
        """
        for _ in range(self.num_of_times):
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=self.n_trials)
            self._export_to_file(study)
            # optuna.visualization.plot_optimization_history(study)
            # optuna.visualization.plot_slice(study)
