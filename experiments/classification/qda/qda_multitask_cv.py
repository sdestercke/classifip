from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
import sys, random, os, csv, numpy as np, pandas as pd
from qda_common import __factory_model, generate_seeds

## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']

from multiprocessing import Process, Queue, cpu_count, JoinableQueue


class ManagerWorkers:

    def __init__(self, nb_process):
        self.workers = None
        self.tasks = Queue()
        self.results = Queue()
        self.qeTraining = [JoinableQueue() for i in range(nb_process)]
        self.NUMBER_OF_PROCESSES = cpu_count() if nb_process is None else nb_process

    def executeAsync(self, model_type, lib_path_server):
        print("starting %d workers" % self.NUMBER_OF_PROCESSES, flush=True)
        self.workers = []
        for i in range(self.NUMBER_OF_PROCESSES):
            p = Process(target=prediction,
                        args=(i, self.tasks, self.qeTraining[i], self.results, model_type, lib_path_server,))
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


def prediction(pid, tasks, queue, results, model_type, lib_path_server):
    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=False)
    while True:
        training = queue.get()
        if training is None: break
        model.learn(**training)
        sum80, sum65 = 0, 0
        while True:
            task = tasks.get()
            if task is None: break
            evaluate, _ = model.evaluate(task['X_test'])
            print("(pid, prediction, ground-truth) ", pid, evaluate, task, flush=True)
            if task['y_test'] in evaluate:
                sum65 += u65(evaluate)
                sum80 += u80(evaluate)
        results.put(dict({'u65': sum65, 'u80': sum80}))
        queue.task_done()
    print("Worker PID finished", pid, flush=True)


def computing_training_testing_step(X_training, y_training, X_testing, y_testing, ell_current,
                                    manager, acc_u65, acc_u80):
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


def computing_best_imprecise_mean(in_path=None, out_path=None, lib_path_server=None, model_type="ilda",
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seed=None, cv_kfold_first=10,
                                  nb_process=2, skip_nfold=0, cv_kfold_second=10, seed_second=None):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_cv", True)
    logger.info('Training dataset (%s, %s, %s)', in_path, out_path, model_type)
    logger.info('Parameters (ells, nbProcess, skip_nfold, cv_kfold_second) (%s, %s, %s, %s, %s, %s)', from_ell,
                to_ell, by_ell, nb_process, skip_nfold, cv_kfold_second)

    data = pd.read_csv(in_path, header=None)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())

    # Seeding a random value for k-fold top learning-testing data
    seed = random.randrange(pow(2, 30)) if seed is None else seed
    logger.debug("[FIRST-STEP-SEED] MODEL: %s, SEED: %s", model_type, seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(model_type, lib_path_server)

    kfFirst = KFold(n_splits=cv_kfold_first, random_state=seed, shuffle=True)
    acc_u80, acc_u65, idx_kfold = dict(), dict(), 0
    seed_2step = generate_seeds(cv_kfold_second) if seed_second is None else seed_second
    logger.debug("[SECOND-STEP-SEEDS] MODEL: %s, SEED: %s, SECOND-SEED: %s", model_type, seed, seed_2step)
    for idx_learning, idx_testing in kfFirst.split(y):
        ell_u65, ell_u80 = dict(), dict()
        # Generate sampling k-fold (learning, testing) for optimal ell parameters
        X_learning, y_learning = X[idx_learning], y[idx_learning]
        X_testing, y_testing = X[idx_testing], y[idx_testing]
        logger.info("Splits %s learning %s", idx_kfold, idx_learning)
        logger.info("Splits %s testing %s", idx_kfold, idx_testing)

        # n-Skipping fold cross-validation (purpose for parallel computing)
        if idx_kfold >= skip_nfold:
            # Generate same k-fold-second (train, test) for impartially computing accuracy all ell parameters
            splits_ell = list([])
            logger.debug("[2-STEP-SEED] MODEL: %s, SEED: %s OF FIRST STEP %s", model_type, seed_2step[idx_kfold], seed)
            kfSecond = KFold(n_splits=cv_kfold_second, random_state=seed_2step[idx_kfold], shuffle=True)
            for idx_learn_train, idx_learn_test in kfSecond.split(y_learning):
                splits_ell.append((idx_learn_train, idx_learn_test))
                logger.info("Splits %s train %s", len(splits_ell), idx_learn_train)
                logger.info("Splits %s test %s", len(splits_ell), idx_learn_test)

            for ell_current in np.arange(from_ell, to_ell, by_ell):
                ell_u65[ell_current], ell_u80[ell_current] = 0, 0
                logger.info("ELL_CURRENT %s", ell_current)
                for idx_learn_train, idx_learn_test in splits_ell:
                    logger.info("Splits step train %s", idx_learn_train)
                    logger.info("Splits step test %s", idx_learn_test)
                    X_cv_train, y_cv_train = X_learning[idx_learn_train], y_learning[idx_learn_train]
                    X_cv_test, y_cv_test = X_learning[idx_learn_test], y_learning[idx_learn_test]

                    ell_u65[ell_current], ell_u80[ell_current] = \
                        computing_training_testing_step(X_cv_train, y_cv_train, X_cv_test, y_cv_test, ell_current,
                                                        manager, ell_u65[ell_current], ell_u80[ell_current])

                    logger.info("Partial-kfold (%s, %s, %s)", ell_current, ell_u65[ell_current], ell_u80[ell_current])
                ell_u65[ell_current] = ell_u65[ell_current] / cv_kfold_first
                ell_u80[ell_current] = ell_u80[ell_current] / cv_kfold_first
                writer.writerow([ell_current, idx_kfold, ell_u65[ell_current], ell_u80[ell_current]])
                file_csv.flush()
                logger.debug("Partial-ell-k-step (%s, %s, %s)", idx_kfold, ell_u65[ell_current], ell_u80[ell_current])
            logger.debug("Total-ell-k-step (%s, %s, %s, %s)", in_path, idx_kfold, ell_u65, ell_u80)

            # Computing optimal ells for using in testing step
            acc_ellu80 = max(ell_u80.values())
            acc_ellu65 = max(ell_u65.values())
            ellu80_opts = [k for k, v in ell_u80.items() if v == acc_ellu80]
            ellu65_opts = [k for k, v in ell_u65.items() if v == acc_ellu65]
            acc_u65[idx_kfold], acc_u80[idx_kfold] = 0, 0
            n_ell80_opts, n_ell65_opts = len(ellu80_opts), len(ellu65_opts)
            for ellu80_opt in ellu80_opts:
                logger.info("ELL_OPTIMAL_CV_U80 %s", ellu80_opt)
                _, acc_u80[idx_kfold] = \
                    computing_training_testing_step(X_learning, y_learning, X_testing, y_testing, ellu80_opt,
                                                    manager, 0, acc_u80[idx_kfold])

            for ellu65_opt in ellu65_opts:
                logger.info("ELL_OPTIMAL_CV_U65 %s", ellu65_opt)
                acc_u65[idx_kfold], _ = \
                    computing_training_testing_step(X_learning, y_learning, X_testing, y_testing, ellu65_opt,
                                                    manager, acc_u65[idx_kfold], 0)

            acc_u65[idx_kfold] = acc_u65[idx_kfold] / n_ell65_opts
            acc_u80[idx_kfold] = acc_u80[idx_kfold] / n_ell80_opts
            writer.writerow([-999, idx_kfold, acc_u65[idx_kfold], acc_u80[idx_kfold]])
            file_csv.flush()
            logger.debug("Partial-ell-2step (u80, u65, accs) (%s, %s, %s, %s, %s)", -999, ellu80_opts, ellu65_opts,
                         acc_u65[idx_kfold], acc_u80[idx_kfold])
        idx_kfold += 1
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Total-accuracy (%s, %s, %s)", in_path, acc_u65, acc_u80)
    logger.debug("Total-avg-accuracy (%s, %s, %s)", in_path, np.mean(list(acc_u65.values())),
                 np.mean(list(acc_u80.values())))


in_path = sys.argv[1]
out_path = sys.argv[2]
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="ilda",
                              from_ell=0.01, to_ell=5.5, by_ell=0.01,  # seed=XXX, skip_nfold=X,
                              lib_path_server=QPBB_PATH_SERVER, nb_process=1)
