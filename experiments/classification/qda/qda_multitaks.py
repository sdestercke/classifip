from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
import sys, random, os, csv, numpy as np, pandas as pd
from qda_common import __factory_model

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
    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=True)
    while True:
        training = queue.get()
        if training is None: break
        model.learn(**training)
        sum80, sum65 = 0, 0
        while True:
            task = tasks.get()
            if task is None: break
            evaluate, _ = model.evaluate(task['X_test'])
            print("(pid, evaluate, task) ", pid, evaluate, task, flush=True)
            if task['y_test'] in evaluate:
                sum65 += u65(evaluate)
                sum80 += u80(evaluate)
        queue.task_done()
        results.put(dict({'u65': sum65, 'u80': sum80}))
    print("Worker PID finished", pid, flush=True)


def computing_best_imprecise_mean(in_path=None, out_path=None, cv_nfold=10, model_type="ieda", test_size=0.4,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seed=None, lib_path_server=None, nb_process=2):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset %s', in_path)
    data = pd.read_csv(in_path)  # , header=None)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())

    ell_u65, ell_u80 = dict(), dict()
    seed = random.randrange(pow(2, 30)) if seed is None else seed
    logger.debug("MODEL: %s, SEED: %s", model_type, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    kf = KFold(n_splits=cv_nfold, random_state=None, shuffle=True)
    splits = list([])
    for idx_train, idx_test in kf.split(y_train):
        splits.append((idx_train, idx_test))
        logger.info("Splits %s train %s", len(splits), idx_train)
        logger.info("Splits %s test %s", len(splits), idx_test)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(model_type, lib_path_server)
    for ell_current in np.arange(from_ell, to_ell, by_ell):
        ell_u65[ell_current], ell_u80[ell_current] = 0, 0
        logger.info("ELL_CURRENT %s", ell_current)
        for idx_train, idx_test in splits:
            logger.info("Splits train %s", idx_train)
            logger.info("Splits test %s", idx_test)
            X_cv_train, y_cv_train = X_train[idx_train], y_train[idx_train]
            X_cv_test, y_cv_test = X_train[idx_test], y_train[idx_test]
            n_test = len(idx_test)

            manager.addNewTraining(X=X_cv_train, y=y_cv_train, ell=ell_current)
            for i, test in enumerate(X_cv_test): manager.addTask({'X_test': test, 'y_test': y_cv_test[i]})
            manager.poisonPillWorkers()

            manager.joinTraining() # wait all process for computing results
            shared_results = manager.getResults()
            shared_results.put('STOP')  ## stop loop queue
            for utility in iter(shared_results.get, 'STOP'):
                ell_u65[ell_current] += utility['u65'] / n_test
                ell_u80[ell_current] += utility['u80'] / n_test
            print("Partial-kfold", ell_current, ell_u65[ell_current], ell_u80[ell_current], flush=True)

        ell_u65[ell_current] = ell_u65[ell_current] / cv_nfold
        ell_u80[ell_current] = ell_u80[ell_current] / cv_nfold
        writer.writerow([ell_current, ell_u65[ell_current], ell_u80[ell_current]])
        file_csv.flush()
        logger.debug("Partial-ell (%s, %s, %s)", ell_current, ell_u65, ell_u80)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Total-ell %s %s %s", in_path, ell_u65, ell_u80)


in_path = sys.argv[1]
out_path = sys.argv[2]
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="ilda",
                              from_ell=0.65, to_ell=1, by_ell=0.01, seed=697720819,
                              lib_path_server=QPBB_PATH_SERVER, nb_process=3)
