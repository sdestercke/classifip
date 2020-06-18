from multiprocessing import Process, Queue, cpu_count, JoinableQueue, Manager
import importlib
from classifip.models.mlc.mlcncc import MLCNCC


def __create_dynamic_class(clazz):
    try:
        st = clazz.split(".")
        module_name = ".".join(st[:-1])
        class_name = st[-1]
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_()
    except Exception as e:
        raise Exception(e, "Creation dynamic class does not complete.")


def prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    try:
        model = __create_dynamic_class(class_model)
        is_compared_with_precise = False
        # if model_challenger is None, thus comparing with precise version
        if class_model_challenger is None:
            class_model_challenger = class_model
            is_compared_with_precise = True
        model_challenger = __create_dynamic_class(class_model_challenger)

        while True:
            # training models
            training = queue.get()
            if training is None:
                break

            MLCNCC.missing_labels_learn_data_set(learn_data_set=training["learn_data_set"],
                                                 nb_labels=training["nb_labels"],
                                                 missing_pct=training["missing_pct"])
            del training["missing_pct"]

            model.learn(**training)
            if class_model_challenger is not None:
                model_challenger.learn(**training)

            while True:
                task = tasks.get()
                if task is None:
                    break
                # prediction of main model
                prediction = model.evaluate(**task['kwargs'])

                # prediction challenger
                prediction_challenger = None
                if class_model_challenger is not None:
                    task['kwargs']['ncc_s_param'] = 0.0 if is_compared_with_precise \
                        else task['kwargs']['ncc_s_param']
                    prediction_challenger = model.evaluate(**task['kwargs'])

                # print and save predictions
                print("(pid, prediction, ground-truth) ", pid,
                      prediction[0] if len(prediction) > 1 else prediction,
                      task['y_test'], flush=True)
                results.append(dict({'prediction': prediction,
                                     'challenger': prediction_challenger,
                                     'ground_truth': task['y_test']}))
            queue.task_done()
    except Exception as e:
        raise Exception(e, "Error in job of PID " + pid)
    finally:
        print("Worker PID finished", pid, flush=True)


class ManagerWorkers:

    def __init__(self, nb_process, fun_prediction=None):
        self.workers = None
        self.tasks = Queue()
        self.manager = Manager()
        self.results = self.manager.list()
        self.qeTraining = [JoinableQueue() for _ in range(nb_process)]
        self.NUMBER_OF_PROCESSES = cpu_count() if nb_process is None else nb_process
        self.fun_prediction = fun_prediction

    def executeAsync(self, class_model, class_model_challenger=None):
        print("Starting %d workers" % self.NUMBER_OF_PROCESSES, flush=True)
        self.workers = []
        for i in range(self.NUMBER_OF_PROCESSES):
            p = Process(target=prediction if self.fun_prediction is None else self.fun_prediction,
                        args=(i, self.tasks, self.qeTraining[i], self.results,
                              class_model, class_model_challenger,))
            self.workers.append(p)

        for w in self.workers:
            w.start()

    def addNewTraining(self, **kwargs):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].put(kwargs)

    def poisonPillTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].put(None)

    def joinTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].join()

    def addTask(self, task):
        self.tasks.put(task)

    def waitWorkers(self):
        for w in self.workers:
            w.join()

    def getResults(self):
        return self.results

    def restartResults(self):
        self.results[:] = []

    def poisonPillWorkers(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.addTask(None)
