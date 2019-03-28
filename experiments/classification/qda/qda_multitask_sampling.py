from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger, normalize_minmax
import sys, random, os, csv, numpy as np, pandas as pd
from qda_common import __factory_model, generate_seeds

## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']

from multiprocessing import Process, Queue, cpu_count, JoinableQueue


class ManagerWorkers:

    def __init__(self, nb_process, criterion="maximality"):
        self.workers = None
        self.tasks = Queue()
        self.results = Queue()
        self.qeTraining = [JoinableQueue() for _ in range(nb_process)]
        self.NUMBER_OF_PROCESSES = cpu_count() if nb_process is None else nb_process
        self.criterion_decision = criterion

    def executeAsync(self, model_type, lib_path_server):
        print("Starting %d workers" % self.NUMBER_OF_PROCESSES, flush=True)
        self.workers = []
        for i in range(self.NUMBER_OF_PROCESSES):
            p = Process(target=prediction,
                        args=(i, self.tasks, self.qeTraining[i], self.results, model_type,
                              lib_path_server, self.criterion_decision))
            self.workers.append(p)

        for w in self.workers:
            w.start()

    def addNewTraining(self, **kwargs):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].put(kwargs)

    def poisonPillTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES): self.qeTraining[i].put(None)

    def joinTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES): self.qeTraining[i].join()

    def addTask(self, task):
        self.tasks.put(task)

    def waitWorkers(self):
        for w in self.workers: w.join()

    def getResults(self):
        return self.results

    def poisonPillWorkers(self):
        for i in range(self.NUMBER_OF_PROCESSES): self.addTask(None)


def prediction(pid, tasks, queue, results, model_type, lib_path_server, criterion="maximality"):
    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=False)
    while True:
        training = queue.get()
        if training is None: break
        model.learn(**training)
        sum80, sum65 = 0, 0
        while True:
            task = tasks.get()
            if task is None: break
            evaluate, _ = model.evaluate(task['X_test'], criterion=criterion)
            print("(pid, prediction, ground-truth, ins)", pid, evaluate, task['y_test'], task['X_test'], flush=True)
            if task['y_test'] in evaluate:
                sum65 += u65(evaluate)
                sum80 += u80(evaluate)
        results.put(dict({'u65': sum65, 'u80': sum80}))
        queue.task_done()
    print("Worker PID finished", pid, flush=True)


def computing_training_testing_step(X_training, y_training, X_testing, y_testing, ell_current, manager, acc_u65,
                                    acc_u80):
    n_test = len(y_testing)
    # Send training data model to every parallel process
    manager.addNewTraining(X=X_training, y=y_training, ell=ell_current)

    # Send testing data to every parallel process
    for i, test in enumerate(X_testing): manager.addTask({'X_test': test, 'y_test': y_testing[i]})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results

    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    shared_results.put('STOP')  ## stop loop queue
    for utility in iter(shared_results.get, 'STOP'):
        acc_u65 += utility['u65'] / n_test
        acc_u80 += utility['u80'] / n_test
    return acc_u65, acc_u80


def performance_cv_accuracy_imprecise(in_path=None, model_type="ilda", ell_optimal=0.1, nb_process=2,
                                      lib_path_server=None, cv_n_fold=10, seeds=None, criterion="maximality"):
    assert os.path.exists(in_path), "Without training data, not testing"
    data = pd.read_csv(in_path)
    logger = create_logger("performance_cv_accuracy_imprecise", True)
    logger.info('Training dataset (%s, %s, %s, %s)', in_path, model_type, ell_optimal, criterion)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    avg_u65, avg_u80 = 0, 0
    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds used for accuracy %s', seeds)
    manager = ManagerWorkers(nb_process=nb_process, criterion=criterion)
    manager.executeAsync(model_type, lib_path_server)
    for time in range(cv_n_fold):
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[time], shuffle=True)
        mean_u65, mean_u80 = 0, 0
        for idx_train, idx_test in kf.split(y):
            logger.info("Splits train %s", idx_train)
            logger.info("Splits test %s", idx_test)
            X_cv_train, y_cv_train = X[idx_train], y[idx_train]
            X_cv_test, y_cv_test = X[idx_test], y[idx_test]
            mean_u65, mean_u80 = computing_training_testing_step(X_cv_train, y_cv_train, X_cv_test, y_cv_test,
                                                                 ell_optimal, manager, mean_u65, mean_u80)
            logger.debug("Partial-kfold (%s, %s, %s, %s)", ell_optimal, time, mean_u65, mean_u80)
        logger.info("Time, seed, u65, u80 (%s, %s, %s, %s)", time, seeds[time],
                    mean_u65 / cv_n_fold, mean_u80 / cv_n_fold)
        avg_u65 += mean_u65 / cv_n_fold
        avg_u80 += mean_u80 / cv_n_fold
    manager.poisonPillTraining()
    logger.debug("total-ell (%s, %s, %s, %s)", in_path, ell_optimal, avg_u65 / cv_n_fold, avg_u80 / cv_n_fold)


def computing_best_imprecise_mean(in_path=None, out_path=None, cv_nfold=10, model_type="ilda", test_size=0.4,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seeds=None, lib_path_server=None,
                                  nb_process=2, n_sampling=10, skip_n_sample=0, criterion="maximality", scaling=False):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_sampling", True)
    logger.info('Training dataset (%s, %s, %s)', in_path, model_type, criterion)
    logger.info('Parameters (size, ells, nbProcess, sampling, nSkip) (%s, %s, %s, %s, %s, %s, %s)', test_size, from_ell,
                to_ell, by_ell, nb_process, n_sampling, skip_n_sample)
    data = pd.read_csv(in_path, header=None)
    X = data.iloc[:, :-1].values
    if scaling: X = normalize_minmax(X)
    y = np.array(data.iloc[:, -1].tolist())

    # Seed for get back up if process is killed
    seeds = generate_seeds(n_sampling) if seeds is None else seeds
    logger.debug("MODEL: %s, SEED: %s", model_type, seeds)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process, criterion=criterion)
    manager.executeAsync(model_type, lib_path_server)
    acc_u80, acc_u65 = dict(), dict()
    for sampling in range(min(n_sampling, len(seeds))):
        X_learning, X_testing, y_learning, y_testing = \
            train_test_split(X, y, test_size=test_size, random_state=seeds[sampling])
        logger.info("Splits %s learning %s", sampling, y_learning)
        logger.info("Splits %s testing %s", sampling, y_testing)

        # n-Skipping sampling and reboot parameter from_ell to 0.01 next sampling
        if skip_n_sample != 0 and sampling > skip_n_sample: from_ell = 0.01
        # n-Skipping sampling testing (purpose for parallel computing)
        if sampling >= skip_n_sample:
            kf = KFold(n_splits=cv_nfold, random_state=None, shuffle=True)
            ell_u65, ell_u80, splits = dict(), dict(), list([])
            for idx_train, idx_test in kf.split(y_learning):
                splits.append((idx_train, idx_test))
                logger.info("Sampling %s Splits %s train %s", sampling, len(splits), idx_train)
                logger.info("Sampling %s Splits %s test %s", sampling, len(splits), idx_test)

            for ell_current in np.arange(from_ell, to_ell, by_ell):
                ell_u65[ell_current], ell_u80[ell_current] = 0, 0
                logger.info("ELL_CURRENT %s", ell_current)
                for idx_train, idx_test in splits:
                    logger.info("Splits train %s", idx_train)
                    logger.info("Splits test %s", idx_test)
                    X_cv_train, y_cv_train = X_learning[idx_train], y_learning[idx_train]
                    X_cv_test, y_cv_test = X_learning[idx_test], y_learning[idx_test]
                    # Computing accuracy testing for cross-validation step
                    ell_u65[ell_current], ell_u80[ell_current] = \
                        computing_training_testing_step(X_cv_train, y_cv_train, X_cv_test, y_cv_test, ell_current,
                                                        manager, ell_u65[ell_current], ell_u80[ell_current])
                    logger.info("Partial-kfold (%s, %s, %s)", ell_current, ell_u65[ell_current], ell_u80[ell_current])

                ell_u65[ell_current] = ell_u65[ell_current] / cv_nfold
                ell_u80[ell_current] = ell_u80[ell_current] / cv_nfold
                writer.writerow([ell_current, sampling, ell_u65[ell_current], ell_u80[ell_current]])
                file_csv.flush()
                logger.debug("Partial-ell-sampling (%s, %s, %s, %s)", ell_current, sampling, ell_u65, ell_u80)
            logger.debug("Total-ell-sampling (%s, %s, %s, %s)", in_path, sampling, ell_u65, ell_u80)

            acc_ellu80 = max(ell_u80.values())
            ellu80_opts = [k for k, v in ell_u80.items() if v == acc_ellu80]

            n_ell_opts = len(ellu80_opts)
            acc_u65[sampling], acc_u80[sampling] = 0, 0
            for ellu80_opt in ellu80_opts:
                logger.info("ELL_OPTIMAL_SELECT_SAMPLING %s", ellu80_opt)
                acc_u65[sampling], acc_u80[sampling] = \
                    computing_training_testing_step(X_learning, y_learning, X_testing, y_testing, ellu80_opt,
                                                    manager, acc_u65[sampling], acc_u80[sampling])
            acc_u65[sampling] = acc_u65[sampling] / n_ell_opts
            acc_u80[sampling] = acc_u80[sampling] / n_ell_opts
            writer.writerow([-999, sampling, acc_u65[sampling], acc_u80[sampling]])
            file_csv.flush()
            logger.debug("Partial-ell-2step (%s, %s, %s, %s)", -999, ellu80_opts, acc_u65[sampling], acc_u80[sampling])

    writer.writerow([-9999, -9, np.mean(list(acc_u65.values())), np.mean(list(acc_u80.values()))])
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Total-accuracy (%s, %s, %s)", in_path, acc_u65, acc_u80)
    logger.debug("Total-avg-accuracy (%s, %s, %s)", in_path, np.mean(list(acc_u65.values())),
                 np.mean(list(acc_u80.values())))


in_path = sys.argv[1]
out_path = sys.argv[2]
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="ilda",
                              from_ell=0.01, to_ell=5.5, by_ell=0.01,  # seeds=XXX, skip_n_sample=X,
                              lib_path_server=QPBB_PATH_SERVER, nb_process=1) #, n_sampling=1)

# in_path = sys.argv[1]
# ell_optimal = float(sys.argv[2])
# QPBB_PATH_SERVER = []  # executed in host
# performance_cv_accuracy_imprecise(in_path=in_path, ell_optimal=ell_optimal, model_type="ilda",
#                                   lib_path_server=QPBB_PATH_SERVER, nb_process=1)
