
import math
import operator
import numpy as np
import itertools

from functools import reduce, partial

from tools import parmap, timeit

def bootstrap(x, y, loss_fun, models, num_samples=200, binary_outcome=True, metrics=None):
    """
    Input:
        x - numpy (n, m) ndarray with samples as rows and features as columns.
        y - numpy column ndarray with value (i) corresponding to the
            sample (i) in x.
        loss_fun - function taking two vertical numpy arrays and returning
            a vertical numpy ndarray of elementwise loss.
        models - list of model fitting procedures taking similarly shaped
            x, y, and loss function. Each procedure returns a function
            'fit': (n, m) numpy ndarray of data "X" -> (n, 1) np array
                "predictions"
        num_samples - number of bootstrap samples
        binary_outcome - True if y is binary. Extremely significantly speeds
            up the bootstrap error calculation.

    Output:
        err - list with a ".632+ bootstrap error" as described by
            Efron, Tibshirani 1997
    """

    def boot_error(fit_model):
        """
        Finds the bootstrap error based on the given data and the model
        fitting procedure.

        Input:
            fit_model - function that takes numpy ndarray x, numpy
                column ndarray y, and loss_fun and returns a function
                'fit': (n, m) numpy ndarray of data "X" -> (n, 1) np array
                "predictions"

        Output:
            bootstrap_error - float ".632+ bootstrap error"
        """

        def one_sample(n):
            """
            Helper function for error. Samples from the data, fits a model,
            and finds the loss on the outsample.

            Output:
                in_test_set - a 1D binary array with 1s indicating the 
                    presence of observation (i) in the test set.

                loss - a 1D array of the loss for observation (i), equal to 0 
                    if (i) was not in the test set.
            """
            in_test_set = np.zeros(n)
            loss = np.zeros(n)

            fit = 0
            while not fit:
                try:
                    train = np.random.choice(n, n)
                    test = np.setdiff1d(np.arange(n), train)

                    fit = fit_model(x[train], y[train])
                except np.linalg.linalg.LinAlgError as e:
                    print("lin_alg_error")
                    pass

            in_test_set[test] = 1
            y_hat = fit(x[test])

            loss[test] = loss_fun(y_hat, y[test])

            return np.concatenate([in_test_set, loss])

        n = len(x)
        q = math.pow(1 - 1/n, n)
        p = 1 - q

        time_sample = lambda x: timeit(lambda:one_sample(n), "Running sample")

        all_samples = reduce(operator.add, 
                             parmap(time_sample, range(num_samples)))
                             # map(time_sample, range(num_samples)))
        in_test_set, loss = np.split(all_samples, 2)

        fit = timeit(lambda:fit_model(x, y), "fitting overall model")
        y_hat = fit(x)
        if metrics is not None: metrics(y_hat, y)

        if any(in_test_set == 0):
            i_in_test_set = lambda i: in_test_set[i] != 0
            good_indices = list(filter(i_in_test_set, range(n)))
            loss = loss[good_indices]
            in_test_set = in_test_set[good_indices]

        err_1 = np.mean(loss/in_test_set)
        err_bar = np.mean(loss_fun(y_hat, y))
        err_632 = q * err_bar + p * err_1

        if binary_outcome:
            p1 = sum(map(lambda t: t == 1, y))/n
            q1 = sum(map(lambda t: t == 1, y_hat))/n
            gamma = p1 * (1 - q1) + q1 * (1 - p1)
        else:
            # gamma = err_bar
            unary_loss = lambda a: loss_fun(*a)
            loss_vector = map(unary_loss, itertools.product(y_hat, y))
            gamma = sum(loss_vector)/n/n
        
        if err_1 > err_bar and gamma > err_bar:
            r = (err_1 - err_bar)/(gamma - err_bar)
        else:
            r = 0

        err1_ = min(err_1, gamma)
        return err_632 + (err1_ - err_bar) * (p * q * r) / (1 - q * r)

    timed_model = lambda model: timeit(lambda: boot_error(model), "model")
    # return np.array(parmap(timed_model, models))
    return list(map(timed_model, models))