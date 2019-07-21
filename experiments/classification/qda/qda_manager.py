from classifip.evaluation.measures import u65, u80
from qda_common import __factory_model
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
                              lib_path_server, self.criterion_decision, ))
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


def prediction(pid, tasks, queue, results, model_type, lib_path_server, criterion):
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
            print("(pid, prediction, ground-truth) (", pid, evaluate, task["y_test"], ")", flush=True)
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
