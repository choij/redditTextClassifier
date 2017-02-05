
import math
import operator
import numpy as np
import itertools

from functools import reduce, partial

from tools import parmap, timeit

def bootstrap(x, y, loss_fun, models, num_samples=200, categorical=True, metrics=None):
    """
    Input:
        x - numpy (n, m) ndarray with samples as rows and features as columns.
        y - numpy column ndarray with value (i) corresponding to the
            sample (i) in x.
        loss_fun - function taking two vertical numpy arrays and returning
            a vertical numpy ndarray of elementwise loss.
        models - list of model objects with 'fit' methods taking similarly 
            shaped x and y. Each procedure has a 'predict' method that takes 
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
            model - object with a function that takes numpy ndarray x, numpy
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

            mod = 0
            while not mod:
                try:
                    train = np.random.choice(n, n)
                    test = np.setdiff1d(np.arange(n), train)

                    mod = model()
                    mod.fit(x[train], y[train])
                except np.linalg.linalg.LinAlgError as e:
                    print("lin_alg_error")
                    pass

            in_test_set[test] = 1
            y_hat = mod.predict(x[test])

            loss[test] = loss_fun(y_hat, y[test])

            return np.concatenate([in_test_set, loss])

        n, _ = x.shape
        q = math.pow(1 - 1/n, n)
        p = 1 - q

        time_sample = lambda x: timeit(lambda:one_sample(n), "Running sample")

        all_samples = reduce(operator.add, 
                             parmap(time_sample, range(num_samples)))
                             
        in_test_set, loss = np.split(all_samples, 2)

        full = model()
        timeit(lambda:full.fit(x, y), "fitting overall model")
        y_hat = full.predict(x)
        if metrics is not None: metrics(y_hat, y)

        if any(in_test_set == 0):
            i_in_test_set = lambda i: in_test_set[i] != 0
            good_indices = list(filter(i_in_test_set, range(n)))
            loss = loss[good_indices]
            in_test_set = in_test_set[good_indices]

        err_1 = np.mean(loss/in_test_set)
        err_bar = np.mean(loss_fun(y_hat, y))
        err_632 = q * err_bar + p * err_1

        if categorical:
            orig_prop = full.priors
            y_au = np.sort(np.vstack((y_hat, full.class_names.reshape(-1,1))))
            i_count = np.asarray(np.unique(y_au, return_counts=True)).T
            counts = np.split(i_count, 2, axis=1)[1].flatten().astype(int) - 1
            obs_prop = counts/counts.sum()

            print(orig_prop, orig_prop.sum())
            print(obs_prop, orig_prop.sum())

            gamma = sum(orig_prop * (1 - obs_prop))

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
    return list(map(timed_model, models))
