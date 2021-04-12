import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from waveml.metrics import RMSE, MSE, MAE, MAPE, MSLE, MBE, SAE, SSE
from waveml.utils import to_tensor, to_array
from sklearn.model_selection import KFold


class Wave:
    def __init__(self, n_opt_rounds: int = 1000, learning_rate: float = 0.01, loss_function=MSE, verbose: int = 1):
        self.n_opt_rounds = int(n_opt_rounds)
        self.learning_rate = float(learning_rate)
        self.loss_function = loss_function
        self.verbose = int(verbose)
        self.fitted = False

        if self.n_opt_rounds < 1:
            raise ValueError(f"n_opt_rounds should belong to an [1;inf) interval, passed {self.n_opt_rounds}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning rate should belong to a (0;inf) interval, passed {self.learning_rate}")
        if self.verbose < 0:
            raise ValueError(f"learning rate should belong to a [0;inf) interval, passed {self.verbose}")


class WaveRegressor(Wave):
    """
    Weighted average regression model
    """

    def __init__(self, n_opt_rounds: int = 1000, learning_rate: float = 0.01, loss_function=MSE, verbose: int = 1):
        super().__init__(n_opt_rounds, learning_rate, loss_function, verbose)

    # Training process
    def fit(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list], weights=None, eval_set=None,
            use_best_model=False) -> None:

        X_train_tensor, y_train_tensor, self.use_best_model = to_tensor(X), to_tensor(y), use_best_model
        self.train_losses, self.test_losses, self.weights_history = [], [], []
        self.fitted = False

        if type(self.use_best_model) != bool:
            raise ValueError(f"use_best_model parameter should be bool, passed {self.use_best_model}")

        self.is_eval_set = True if eval_set != None else False
        if self.is_eval_set:
            X_test_tensor = to_tensor(eval_set[0])
            y_test_tensor = to_tensor(eval_set[1])

        n_features = X_train_tensor.shape[1]

        self.weights = to_tensor(weights) if weights != None else torch.tensor(
            [1 / n_features for i in range(n_features)]
        )

        self.weights.requires_grad_()
        self.optimizer = torch.optim.Adam([self.weights], self.learning_rate)

        for i in range(self.n_opt_rounds):
            # clear gradient
            self.optimizer.zero_grad()
            # get train set error
            train_loss = self.__opt_func(X_segment=X_train_tensor, y_segment=y_train_tensor)
            # append train loss to train loss history
            self.train_losses.append(train_loss.item())
            # create a train part of fit information
            train_output = f"train: {train_loss.item()}"
            # optimization of weights according to the function
            train_loss.backward()

            # create a test part of fit information
            test_output = ""
            if self.is_eval_set:
                # get test set error
                test_loss = self.__opt_func(X_segment=X_test_tensor, y_segment=y_test_tensor)
                # append test loss to test loss history
                self.test_losses.append(test_loss.item())
                test_output = f"test: {test_loss.item()}"

            if self.verbose != 0:
                print(f"round: {i}", train_output, test_output)
            self.weights_history.append(self.weights)

            self.optimizer.step()

        self.fitted = True

    # Get a tensor of weights after training
    def get_weights(self) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        if self.use_best_model:
            return self.weights_history[self.test_losses.index(min(self.test_losses))].detach().numpy()
        return self.weights_history[self.train_losses.index(min(self.train_losses))].detach().numpy()

    # Predict on on passed data with current weights
    def predict(self, X) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X = to_tensor(X)
        sum = torch.sum(X * self.get_weights(), 1)
        return sum.detach().numpy()

    def score(self, X_train, y_test):
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X_train_tensor, y_test_tensor = to_tensor(X_train), to_tensor(y_test)
        y_pred = self.predict(X_train_tensor)
        return self.loss_function(y_test_tensor, y_pred).item()

    def plot(self) -> None:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        plt.plot([i for i in range(self.n_opt_rounds)], self.train_losses)
        if self.is_eval_set:
            plt.plot([i for i in range(self.n_opt_rounds)], self.test_losses)
        plt.show()
        return

    # Function for weight optimization
    def __opt_func(self, X_segment, y_segment):
        y_true = y_segment
        y_pred = self.__inner_predict(X_segment)
        return self.loss_function(y_true, y_pred)

    def __inner_predict(self, X) -> torch.tensor:
        sum = torch.sum(X * self.weights, 1)
        return sum


class WaveClassifier(WaveRegressor):
    """
    Weighted average classification model
    """

    def __init__(self, n_opt_rounds: int = 1000, learning_rate: float = 0.01, loss_function=MSE, verbose: int = 1,
                 threshold: float = 0.5):
        super().__init__(n_opt_rounds, learning_rate, loss_function, verbose)
        self.threshold = threshold

    # Predict on on passed data with current weights
    def predict_proba(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X = to_tensor(X)
        proba = (1 / (1 + torch.exp(-torch.sum(X * self.weights, 1)))).detach().numpy().reshape(-1, 1)
        inverse_proba = -proba + 1
        return np.hstack([inverse_proba, proba])

    def predict(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        y_pred = self.predict_proba(X)[:, 1]
        y_pred[y_pred < self.threshold] = 0
        y_pred[y_pred >= self.threshold] = 1
        return y_pred


class WaveTransformer(Wave):
    """
    Weighted average transformer, which performs a transformation over each feature separately
    """

    def __init__(self, n_opt_rounds: int = 1000, learning_rate: float = 0.01, loss_function=MSE,
                 regression: bool = True,
                 verbose: int = 1, n_folds: int = 4, random_state: int = None, shuffle: bool = False):
        super().__init__(n_opt_rounds, learning_rate, loss_function, verbose)

        self.regression = regression
        self.shuffle = shuffle
        self.n_folds = n_folds
        self.random_state = random_state

        if n_folds < 2:
            raise ValueError(f"n_folds should belong to a [2;inf) interval, passed {self.verbose}")
        if n_folds < 0:
            raise ValueError(f"random_state should belong to a [0;inf) interval, passed {self.verbose}")
        self.n_folds = n_folds

        self.shuffle = shuffle
        self.random_state = int(random_state) if self.shuffle else None

    def __opt_func(self, X_segment, y_segment, weights):
        return self.loss_function(X_segment * weights[0] + weights[1], y_segment)

    def fit(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]):

        X_train_tensor, y_train_tensor = to_tensor(X), to_tensor(y)

        self.n_features = X_train_tensor.shape[1]
        self.weights = []

        self.fitted = False

        for i in range(self.n_features):
            feature_weights = torch.tensor([])

            X = X_train_tensor[:, i]
            kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle)

            print("\nFeature:", i)
            f = 0
            for train_index, test_index in kf.split(X):
                fold_weights = torch.tensor([1.0, 0.0])
                fold_weights.requires_grad_()
                self.optimizer = torch.optim.Adam([fold_weights], self.learning_rate)

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_train_tensor[train_index], y_train_tensor[test_index]

                for j in range(self.n_opt_rounds):
                    self.optimizer.zero_grad()
                    # get train set error
                    train_loss = self.__opt_func(X_segment=X_train, y_segment=y_train, weights=fold_weights)
                    # create a train part of fit information
                    train_output = f"train: {train_loss.item()}"
                    # optimization of weights according to the function
                    if self.verbose >= 1:
                        print("round:", j, train_output)
                    train_loss.backward()
                    self.optimizer.step()

                if self.verbose in [1, 2]:
                    print(f"\tFold {f}:",
                          self.__opt_func(X_segment=X_test, y_segment=y_test, weights=fold_weights).item())
                f += 1
                feature_weights = torch.cat([feature_weights, fold_weights])
            feature_weights = feature_weights.reshape(-1, 2)
            self.weights.append(feature_weights)
        self.fitted = True
        return self

    def get_weights(self) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        return torch.tensor(self.weights).detach().numpy()

    def transform(self, X) -> np.ndarray:
        X_tensor = to_tensor(X)

        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        for i in range(self.n_features):
            feature = X_tensor[:, i]
            w = self.weights[i].mean(dim=0)
            if self.regression:
                X_tensor[:, i] = feature * w[0] + w[1]
            else:
                (1 / (1 + torch.exp(-(feature * w[0] + w[1]))))

        return X_tensor.detach().numpy()

    def fit_transform(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
                      y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class WaveEncoder:
    """
    Categorical feature incoding model
    """

    def __init__(self, encodeing_type: str, strategy: str = "mean"):
        self.encoding_types = ["catboost", "label", "target", "count"]
        self.encoding_type = encodeing_type.lower()
        if self.encoding_type not in self.encoding_types:
            raise ValueError(f"Given encodint type {self.encoding_type}, allowed {', '.join(self.encoding_types)}")

        self.strategy = strategy
        if self.strategy != None:
            self.strategies = {"mean": np.mean,
                               "median": np.median,
                               "sum": np.sum}

            self.strategy = self.strategy.lower()
            if self.strategy not in list(self.strategies.keys()):
                raise ValueError(
                    f"Given strategy type {self.strategy}, allowed {', '.join(list(self.strategies.keys()))}")

        self.fitted = False

    def fit(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list] = None, regression=True,
            cat_features=None) -> None:

        self.fitted = False
        self.regression = regression
        self.X = to_array(X)
        self.y = to_array(y) if type(y) != None else None

        self.n_features = X.shape[1]

        self.cat_features = cat_features
        if self.cat_features == None:
            self.cat_features = [i for i in range(self.n_features)]

        if self.encoding_type == "count":
            self.__count(self.X)

        if self.encoding_type == "target":
            self.__target(self.X, self.y)

        if self.encoding_type == "label":
            self.__label(self.X)

    def __count(self, X: np.ndarray) -> None:
        self.mappers = []
        for i in range(len(self.cat_features)):
            counts = dict()
            feature = X[:, self.cat_features[i]]
            uniques = np.unique(feature)

            for unique in uniques:
                counts[unique] = len(feature[feature == unique])

            self.mappers.append(counts)

    def __target(self, X: np.ndarray, y: np.ndarray) -> None:
        X_labeled = self.__label(X)
        if self.regression:
            self.mappers = []
            for i in range(len(self.cat_features)):
                counts = dict()
                feature = X_labeled[:, self.cat_features[i]]
                uniques = np.unique(feature)

                for unique in uniques:
                    counts[unique] = self.strategies.get(self.strategy)(y[feature == unique])

                self.mappers.append(counts)
        else:
            self.mappers = []
            for i in range(len(self.cat_features)):
                counts = dict()
                feature = X[:, self.cat_features[i]]
                uniques = np.unique(feature)

                for unique in uniques:
                    counts[unique] = y[feature == unique].mean()

                self.mappers.append(counts)

    def __label(self, X: np.ndarray) -> None:
        self.mappers = []
        for i in range(len(self.cat_features)):
            counts = dict()
            feature = X[:, self.cat_features[i]]
            uniques = np.unique(feature)

            for i in range(len(uniques)):
                counts[uniques[i]] = float(i)

            self.mappers.append(counts)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_array = to_array(X)
        n_features = X_array.shape[1]
        if self.n_features != n_features:
            raise ValueError(
                f"Shape of fitting array ({self.n_features} columns) and transforming array ({n_features} columns) do not match")

        for i in range(len(self.cat_features)):
            X_array[:, self.cat_features[i]] = np.vectorize(self.mappers[i].get)(X_array[:, self.cat_features[i]])

        return X_array

    def fit_transform(self, X, y=None, regression=True, cat_features=None) -> np.ndarray:
        self.fit(X, y, regression, cat_features)
        return self.transform(X)

    # ADD: __catboost


class WaveStackingTransformer:
    """
    Stacking ensembling model
    """

    def __init__(self, models, metric, n_folds: int = 4, random_state: int = None, shuffle: bool = False,
                 verbose: bool = True, regression: bool = True, voting: str = "soft"):

        self.models = [i[1] for i in models]
        self.n_models = len(self.models)
        self.names = [i[0] for i in models]
        self.model_dict = {}
        self.model_scores_dict = {}

        self.n_folds = n_folds
        self.metric = metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.regression = regression
        self.soft_voting = False
        self.hard_voting = False

        if not self.regression:
            if voting.lower() == "soft":
                self.soft_voting = True
            elif voting.lower() == "hard":
                self.hard_voting = True
            else:
                raise ValueError(f'Unknown voting type: {voting}, expected "soft" or "gard"')

        self.fitted = False

    def fit(self,
            X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]):

        self.model_dict = {}
        self.model_scores_dict = {}

        self.X, self.y = to_array(X), to_array(y)

        if self.soft_voting:
            self.n_labels = len(np.unique(y))

        kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle)

        # Form each model
        for i in range(self.n_models):
            # For each fold
            sub_models = []
            sub_scores = []
            fold = 0
            for train_index, test_index in kf.split(X):
                sub_model = self.models[i]

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                sub_model.fit(X_train, y_train)
                sub_models.append(sub_model)

                if self.regression or self.hard_voting:
                    sub_score = self.metric(sub_model.predict(X_test), y_test)
                else:
                    sub_score = self.metric(sub_model.predict_proba(X_test), y_test)

                sub_scores.append(sub_score)

                if self.verbose:
                    print(f"Fold: {fold}, Score: {sub_score}")

                fold += 1

            self.model_dict[self.names[i]] = sub_models
            self.model_scores_dict[self.names[i]] = sub_scores

        self.fitted = True
        return self

    def transform(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
                  prettified: bool = False) -> [np.ndarray, pd.DataFrame]:

        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        X = to_array(X)
        transformations = np.zeros([len(X), 1])

        for key in list(self.model_dict.keys()):
            sub_transformations = np.zeros([len(X), 1])

            sub_models = self.model_dict.get(key)

            for sub_model in sub_models:
                if self.regression or self.hard_voting:
                    sub_transformations = np.hstack((sub_transformations, sub_model.predict(X).reshape(-1, 1)))
                else:
                    sub_transformations = np.hstack(
                        (sub_transformations, sub_model.predict_proba(X)[:, 1].reshape(-1, 1)))

            sub_transformations = sub_transformations[:, 1:]

            if self.regression or self.soft_voting:
                transformations = np.hstack((transformations, sub_transformations.mean(axis=1).reshape(-1, 1)))
            else:
                def voting(x):
                    x = list(x)
                    if x.count(0) > x.count(1):
                        return 0
                    else:
                        return 1

                transformations = np.hstack(
                    [transformations, np.array(list(map(voting, sub_transformations))).reshape(-1, 1)])

        transformations = transformations[:, 1:]
        if prettified:
            return pd.DataFrame(columns=self.names, data=transformations)
        return transformations

    def fit_transform(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
                      y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list], prettified: bool = False) -> [
        np.ndarray,
        pd.DataFrame]:

        self.model_dict = {}
        self.model_scores_dict = {}

        self.X, self.y = to_array(X), to_array(y)

        kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle)

        # Form each model
        transformations = np.zeros([len(X), self.n_models])

        for i in range(self.n_models):
            # For each fold
            sub_models = []
            sub_scores = []
            fold = 0
            sub_transformations = np.zeros([len(X), 1])

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                sub_model = self.models[i]
                sub_model.fit(X_train, y_train)

                if self.regression or self.hard_voting:
                    sub_transformations[test_index] = sub_model.predict(X_test).reshape(-1, 1)
                else:
                    sub_transformations[test_index] = sub_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

                sub_models.append(sub_model)
                sub_score = self.metric(sub_model.predict(X_test), y_test)
                sub_scores.append(sub_score)

                if self.verbose:
                    print(f"Fold: {fold}, Score: {sub_score}")

                fold += 1

            transformations[:, i] = sub_transformations[:, 0]

            self.model_dict[self.names[i]] = sub_models
            self.model_scores_dict[self.names[i]] = sub_scores

        self.fitted = True

        if prettified:
            return pd.DataFrame(columns=self.names, data=transformations)

        return transformations

    def get_scores(self, prettified: bool = False) -> [np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        if prettified:
            return pd.DataFrame(data=self.model_scores_dict)

        return np.array(list(self.model_scores_dict.values()))

    def get_models(self) -> [np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise AttributeError("Model has not been fitted yet. Use fit() method first.")

        return self.model_dict.values()


class WaveCompensator:
    def __init__(self, strong_model, weak_model, n_folds: int = 4, random_state: int = None,
                 shuffle: bool = False, n_opt_rounds: int = 1000, learning_rate: float = 0.01,
                 regression: bool = True, loss_function=MSE, verbose: bool = True):

        self.strong_model, self.weak_model = strong_model, weak_model
        self.strong_models, self.weak_models = [], []
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.n_opt_rounds = n_opt_rounds
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.regression = regression
        self.verbose = verbose

        self.train_losses = []

        self.fitted = False

    def fit(self,
            X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> None:

        self.strong_models, self.weak_models = [], []
        self.X, self.y = to_array(X), to_array(y)

        kf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=self.shuffle)

        strong_train = np.zeros([len(X), 1])
        weak_train = np.zeros([len(X), 1])

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.strong_model.fit(X_train, y_train)
            self.weak_model.fit(X_train, y_train)

            self.strong_models.append(self.strong_model)
            self.weak_models.append(self.weak_model)

            strong_train[test_index] = self.strong_model.predict(X_test).reshape(-1, 1)
            weak_train[test_index] = self.weak_model.predict(X_test).reshape(-1, 1)

        self.weight = torch.tensor([0.0])
        self.weight.requires_grad_()
        self.optimizer = torch.optim.SGD([self.weight], self.learning_rate)

        for i in range(self.n_opt_rounds):
            # clear gradient
            self.optimizer.zero_grad()
            # get train set error
            train_loss = self.__opt_func(X_strong=to_tensor(strong_train), X_weak=to_tensor(weak_train), y=to_tensor(y))
            # append train loss to train loss history
            self.train_losses.append(train_loss.item())
            # create a train part of fit information
            print(f"train: {train_loss.item()}")
            # optimization of weights according to the function
            train_loss.backward()
            self.optimizer.step()

        self.fitted = True

    def predict(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        X = to_array(X)
        strong_preds = np.zeros(len(X))
        weak_preds = np.zeros(len(X))

        for i in range(self.n_folds):
            strong_preds += (self.strong_models[i].predict(X)) / self.n_folds
            weak_preds += (self.weak_models[i].predict(X)) / self.n_folds

        return strong_preds - self.weight.item() * weak_preds

    def predict_proba(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        X = to_array(X)
        strong_preds = np.zeros((len(X), 1))
        weak_preds = np.zeros((len(X), 1))

        for i in range(self.n_folds):
            if self.regression:
                strong_preds += self.strong_models[i].predict_proba(X)[:, 1].reshape(-1, 1) / self.n_folds
                weak_preds += self.weak_models[i].predict_proba(X)[:, 1].reshape(-1, 1) / self.n_folds

        proba = strong_preds - self.weight.item() * weak_preds
        inverse_proba = -proba + 1
        return np.hstack([inverse_proba, proba])

    def __opt_func(self, X_strong, X_weak, y) -> torch.Tensor:
        return self.loss_function(X_strong - self.weight * X_weak, y)


class WaveBagging:
    def __init__(self, model, n_estimators: int = 10, bagging_percentage: float = 0.7, regression: bool = True,
                 verbose: int = 1):
        self.model = model
        self.n_estimators = n_estimators
        self.bagging_percentage = bagging_percentage
        self.regression = regression
        self.verbose = verbose
        self.trained_models = []

    def fit(self,
            X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list],
            y: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> None:

        X, y = to_array(X), to_array(y)
        self.trained_models = []
        len_rows = X.shape[0]
        n_bag_items = round(len_rows * self.bagging_percentage)

        for i in range(self.n_estimators):
            random_indices = np.random.choice(len_rows, size=n_bag_items, replace=False)
            X_bag, y_bag = X[random_indices], y[random_indices]
            self.model.fit(X_bag, y_bag)
            self.trained_models.append(self.model)
            print(f"Model {i} is fitted")

    def predict(self, X: [pd.DataFrame, pd.Series, np.array, torch.Tensor, list]) -> np.ndarray:
        X = to_array(X)
        preds = []
        for model in self.trained_models:
            if self.regression:
                pred = model.predict(X).reshape(-1, 1)
                preds.append(pred)
            else:
                pred = model.predict_proba(X)[:, 1].reshape(-1, 1)
                preds.append(pred)

        y_pred = np.hstack(preds).mean(axis=1)
        return y_pred
