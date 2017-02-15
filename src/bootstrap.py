
import math
import operator
import numpy as np
import itertools
import pickle

from functools import reduce, partial
from metrics import CategoricalMetric
from tools import parmap, timeit, timit

class Bootstrap:
    """
    Note: multiprocessing doesn't pass objects properly to update the metrics,
        so currently only using a single process.
    """

    def __init__(self, x, y, models, loss_fun=lambda y_hat, y: np.mean(y_hat==y),num_samples=200, categorical=True):
        self.x = x
        self.y = y
        self.loss_fun = loss_fun
        self.models = models
        self.num_samples = num_samples
        self.categorical = categorical

        self.n = x.shape[0]
        self.estimates = []

    def print_summary(self):
        for i in range(len(self.models)):
            self.models[i].print_metrics()
            text = "Bootstrap .632+ error estimate: {:.5f}"
            print(text.format(self.estimates[i]))

    def comma_separated_metrics(self, prepend=lambda x: []):
        for i in range(len(self.models)):
            metrics = self.models[i].get_metrics()
            metrics.append(self.estimates[i])
            metrics = list(map(str, metrics))
            metrics = prepend(self.models[i]) + metrics
            print(", ".join(metrics))

    # @timit
    def run(self):
        """
        Input:
            x - numpy (n, m) ndarray with samples as rows and features as 
                columns.
            y - numpy column ndarray with value (i) corresponding to the
                sample (i) in x.
            loss_fun - function taking two vertical numpy arrays and returning
                a vertical numpy ndarray of elementwise loss.
            models - list of model objects with 'fit' methods taking 
                similarly shaped x and y. Each procedure has a 'predict'
                method that takes 
                another x and returns a vector y.
            num_samples - number of bootstrap samples
            catigorical - True if y is categorical. Extremely significant 
                speed-up for the bootstrap error calculation.

        Output:
            err - list with a ".632+ bootstrap error" as described by
                Efron, Tibshirani 1997
        """

        def boot_error(model):
            """
            Finds the bootstrap error based on the given data and the model
            fitting procedure.

            Input:
                model - object with a function that takes numpy ndarray x, 
                    numpy column ndarray y, and loss_fun and returns a 
                    function 'fit': (n, m) numpy ndarray of data 
                    "X" -> (n, 1) np array "predictions"

            Output:
                bootstrap_error - float ".632+ bootstrap error"
            """
            # @timit
            def one_sample():
                """
                Helper function for error. Samples from the data, fits a model,
                and finds the loss on the outsample.

                Output:
                    in_test_set - a 1D binary array with 1s indicating the 
                        presence of observation (i) in the test set.

                    loss - a 1D array of the loss for observation (i), equal to 0 
                        if (i) was not in the test set.
                """
                in_test_set = np.zeros(self.n)
                loss = np.zeros(self.n)

                model.unfit()
                while not model.is_fitted():
                    try:
                        train = np.random.choice(self.n, self.n)
                        test = np.setdiff1d(np.arange(self.n), train)

                        model.fit(self.x[train], self.y[train])
                    except np.linalg.linalg.LinAlgError as e:
                        print("lin_alg_error")
                        pass

                in_test_set[test] = 1
                y_hat = model.predict(self.x[test])
                model.update_metrics(y_hat, self.y[test])

                loss[test] = self.loss_fun(y_hat, self.y[test])

                return np.concatenate([in_test_set, loss])

            def run_samples():

                # all_samples = parmap(one_sample, range(self.num_samples))
                all_samples = [one_sample() for i in range(self.num_samples)]

                return np.split(reduce(operator.add, all_samples), 2)

            q = math.pow(1 - 1/self.n, self.n)
            p = 1 - q

            in_test_set, loss = run_samples()
            
            model.fit(self.x, self.y)
            y_hat = model.predict(self.x)

            if any(in_test_set == 0):
                i_in_test_set = lambda i: in_test_set[i] != 0
                good_indices = list(filter(i_in_test_set, range(self.n)))
                loss = loss[good_indices]
                in_test_set = in_test_set[good_indices]

            err_1 = np.mean(loss/in_test_set)
            err_bar = np.mean(self.loss_fun(y_hat, self.y))
            err_632 = q * err_bar + p * err_1

            if self.categorical:
                class_names = set(np.unique(self.y))
                counts = {c:0 for c in class_names}
                for i in range(len(self.y)):
                    counts[self.y[i][0]] += 1
                orig_prop = np.array([counts[c] for c in sorted(class_names)])
                vert_names = model.class_names.reshape(-1,1)
                y_au = np.sort(np.vstack((y_hat, vert_names)))
                i_count = np.asarray(np.unique(y_au, return_counts=True)).T
                counts = np.split(i_count, 2, axis=1)[1].flatten().astype(int)
                counts -= 1 # remove effect of adding all class names
                obs_prop = counts/counts.sum()

                gamma = (orig_prop * (1 - obs_prop)).sum()

            else:
                # gamma = err_bar
                unary_loss = lambda a: self.loss_fun(*a)
                all_pairs = itertools.product(y_hat, self.y)
                loss_vector = map(unary_loss, all_pairs)
                gamma = sum(loss_vector)/self.n/self.n
            
            if err_1 > err_bar and gamma > err_bar:
                r = (err_1 - err_bar)/(gamma - err_bar)
            else:
                r = 0

            err1_ = min(err_1, gamma)
            return err_632 + (err1_ - err_bar) * (p * q * r) / (1 - q * r)
  
        self.estimates = list(map(boot_error, self.models))

class BootstrapModel:
    def __init__(self, filepath='no_filepath_set'):
        self.fitted = False
        self.metrics = CategoricalMetric()
        self.most_recent_y_hat = None
        self.filepath = filepath

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def fit(self, x, y):
        pass

    def is_fitted(self):
        return self.fitted

    def unfit(self):
        self.fitted = False

    def update_metrics(self, y_hat, y_test):
        self.metrics.update(y_hat, y_test)

    def print_metrics(self):
        self.metrics.print()

    def get_metrics(self):
        return self.metrics.get_metrics()