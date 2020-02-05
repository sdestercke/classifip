from classifip.evaluation.measures import hamming_distance
from classifip.representations import binaryTree as bt
from classifip.representations.intervalsProbability import IntervalsProbability
from itertools import permutations, combinations, product
import queue, copy, math, abc, time
from classifip.utils import timeit
import numpy as np
from .mlcncc import MLCNCC
from classifip.utils import create_logger
from itertools import compress


class BinaryMultiLabel(bt.BinaryTree):

    # @timeit
    def getlowerexpectation(self, function, item, ncc_s_param=2, ncc_epsilon=0.001):

        class_values = self.node.label
        nb_class = len(class_values)

        if function.shape != (nb_class,):
            raise Exception('Size of cost vector is not correct:', function.shape)
        cost = function

        def lowerExp(NDtree):
            bound_probabilities = NDtree.node.proba(item, ncc_s_param, ncc_epsilon)
            if bound_probabilities.isreachable() == 0:
                bound_probabilities.setreachableprobability()

            if NDtree.left.node.count() == 1 and NDtree.left.node.count() == 1:
                expInf = bound_probabilities.getlowerexpectation(
                    function=np.array([cost[class_values.index(NDtree.left.node.label[0])],
                                       cost[class_values.index(NDtree.right.node.label[0])]]))

            elif NDtree.left.node.count() == 1:
                expInf = bound_probabilities.getlowerexpectation(
                    function=np.array([cost[class_values.index(NDtree.left.node.label[0])],
                                       lowerExp(NDtree.right)]))

            elif NDtree.right.node.count() == 1:
                expInf = bound_probabilities.getlowerexpectation(
                    function=np.array([lowerExp(NDtree.left),
                                       cost[class_values.index(NDtree.left.node.label[0])]]))

            else:
                expInf = bound_probabilities.getlowerexpectation(
                    function=np.array([lowerExp(NDtree.left),
                                       lowerExp(NDtree.right)]))
            return expInf

        return lowerExp(self)


class MLCNCCExact(MLCNCC):

    def __init__(self, DEBUG=False):
        super(MLCNCCExact, self).__init__()
        self.power_set = []
        self.root = None
        self.DEBUG = DEBUG
        self._logger = create_logger("MLCNCCExact", DEBUG)

    def learn(self, learn_data_set, nb_labels=1, seed_random_label=None):
        super(MLCNCCExact, self).learn(learn_data_set, nb_labels, seed_random_label)

        label_prior = [lab_count / float(self.training_size) for lab_count in self.label_counts]
        self.power_set = ["".join(seq) for seq in product("01", repeat=self.nb_labels)]
        self.root = BinaryMultiLabel(label=self.power_set)
        tbinary = queue.LifoQueue()
        tbinary.put(self.root)
        while not tbinary.empty():
            tree_recursive = tbinary.get()
            sub_power_set = tree_recursive.node.label[:int(tree_recursive.node.count() / 2)]
            left, right = tree_recursive.node.splitNode(label=sub_power_set)
            tree_recursive.left = bt.BinaryTree(node=left)
            tree_recursive.right = bt.BinaryTree(node=right)
            tree_recursive.node.proba = self.__learning_sub_model(tree_recursive.node.label, label_prior)
            if len(sub_power_set) > 1:
                tbinary.put(tree_recursive.right)
                tbinary.put(tree_recursive.left)

    def __learning_sub_model(self, labels, label_prior):
        level_tree_model = int(self.nb_labels - math.log2(len(labels)))
        augmented_input_labels = [str(lab) for lab in labels[0][:level_tree_model]]

        def __inference(item, ncc_s_param=2, ncc_epsilon=0.001):
            u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                super(MLCNCCExact, self).lower_upper_cond_probability(level_tree_model, label_prior, item,
                                                                      augmented_input_labels,
                                                                      ncc_s_param, ncc_epsilon)
            y1_upper = u_numerator_1 / (u_numerator_1 + l_denominator_0)
            y1_lower = l_numerator_1 / (l_numerator_1 + u_denominator_0)
            y0_upper = u_denominator_0 / (u_denominator_0 + l_numerator_1)
            y0_lower = l_denominator_0 / (l_denominator_0 + u_numerator_1)
            interval_probability = np.array([[y0_upper, y1_upper], [y0_lower, y1_lower]])
            return IntervalsProbability(interval_probability)

        return __inference

    def __evaluate_single_instance(self, new_instance, ncc_s_param=2, ncc_epsilon=0.001):
        """
            It only works for a single new instance
        :param new_instance:
        :param ncc_s_param:
        :param ncc_epsilon:
        :return:
        """
        indices_labels = list(range(self.nb_labels))
        all_output_space = list(product([0, 1], repeat=self.nb_labels))
        maximality_sets = dict.fromkeys(all_output_space, True)
        PARTIAL_VALUE = self.LABEL_PARTIAL_VALUE
        __logger = self._logger

        # @timeit
        def calculation_cost(a_sub_vector, neq_idx_labels=None):
            _cost_vector = []
            for y in all_output_space:
                y_sub_array = np.array(y)[neq_idx_labels] if neq_idx_labels is not None else y
                _cost_vector.append(hamming_distance(y_sub_array, a_sub_vector))
            return np.array(_cost_vector)

        # @timeit
        def expansion_partial_binary_vector(binary_vector):
            set_binary_vector = []
            set_dominated_vector = []
            current = np.array([], dtype=int)
            q_current = queue.Queue()
            partial_index = np.ones(len(binary_vector), dtype=bool)
            for idx, label in enumerate(binary_vector):
                q_next = queue.Queue()
                if not q_current.empty():
                    while not q_current.empty():
                        current = q_current.get()
                        if label == 3:
                            partial_index[idx] = False
                            q_next.put(np.append(current, 1))
                            q_next.put(np.append(current, 0))
                        else:
                            q_next.put(np.append(current, label))
                else:
                    if label == 3:
                        partial_index[idx] = False
                        q_next.put(np.append(current, 1))
                        q_next.put(np.append(current, 0))
                    else:
                        q_next.put(np.append(current, label))
                q_current = q_next

            while not q_current.empty():
                solution = q_current.get()
                neg_solution = np.array(solution)
                neg_solution[partial_index] = 1 - neg_solution[partial_index]
                set_binary_vector.append(tuple(solution))
                set_dominated_vector.append(tuple(neg_solution))

            return set_binary_vector, set_dominated_vector

        # @timeit
        def is_solution_not_dominated(partial_prediction):
            if PARTIAL_VALUE in partial_prediction:
                set_dominant, set_dominated = expansion_partial_binary_vector(partial_prediction)
            else:
                set_dominant, set_dominated = [tuple(partial_prediction)], [tuple(1 - partial_prediction)]
            is_not_dominated = False
            for idx in range(len(set_dominant)):
                if maximality_sets[set_dominant[idx]] and maximality_sets[set_dominated[idx]]:
                    is_not_dominated = True
                    break
            return is_not_dominated, set_dominated

        # @timeit
        def mark_solution_dominated(set_dominated_preds):
            for dominated in set_dominated_preds:
                maximality_sets[dominated] = False

        # @timeit
        def inf_not_equal_labels(root, nb_labels, p_n_not_equal_indices, p_neq_idx=None):
            for a_vector in product([0, 1], repeat=p_n_not_equal_indices):
                partial_prediction = PARTIAL_VALUE * np.ones(nb_labels, dtype=int)
                partial_prediction[p_neq_idx] = np.array(a_vector)
                is_not_dominated, set_dominated_preds = is_solution_not_dominated(partial_prediction)
                if is_not_dominated:
                    not_a_vector = 1 - np.array(a_vector)
                    cost_vector = calculation_cost(not_a_vector, p_neq_idx)
                    inf_expectation = root.getlowerexpectation(cost_vector, new_instance, ncc_s_param, ncc_epsilon)
                    __logger.debug("%s >_M %s ==> %s <? %s", a_vector, not_a_vector,
                                   (p_n_not_equal_indices * 0.5), inf_expectation)
                    if (p_n_not_equal_indices * 0.5) < inf_expectation:
                        mark_solution_dominated(set_dominated_preds)

        # some equal labels in comparison (m1 > m2)
        for i in range(self.nb_labels - 1):
            n_not_equal_indices = (self.nb_labels - 1) - i
            a_set_indices = list(combinations(indices_labels, n_not_equal_indices))
            for neq_idx in a_set_indices:
                inf_not_equal_labels(self.root, self.nb_labels, n_not_equal_indices, list(neq_idx))
        # none equal labels (all different)
        inf_not_equal_labels(self.root, self.nb_labels, self.nb_labels)

        solution_exact = list(filter(lambda k: maximality_sets[k], maximality_sets.keys()))
        self._logger.debug("set solutions improved exact inference %s", solution_exact)
        return solution_exact

    def evaluate(self, test_dataset, ncc_s_param=2, ncc_epsilon=0.001):
        solutions = []
        for item in test_dataset:
            start = time.time()
            solutions.append(self.__evaluate_single_instance(item, ncc_s_param, ncc_epsilon))
            self._logger.debug("Time-Inference-Instance %s ", (time.time() - start))
            if self.DEBUG:
                self._logger.debug("Tree-probabilities")
                self.root.printProba(item)
        return solutions

    def evaluate_exact(self, test_dataset, ncc_s_param=2, ncc_epsilon=0.001):
        """
            This algorithm use the criterion of maximality used in CredalSet class
            it exploits the transitivity property checking only dominant class with
            uncheck class, e.g.
                If a > b (a dominates b) and if b dominates c, then it is not necessary
                to check b > c, because the expectation of a > c should be positive par
                transitivity, in other words, a dominates c (this last is checked)
        :param test_dataset:
        :param ncc_s_param:
        :param ncc_epsilon:
        :return:
        """
        all_output_space = list(product([0, 1], repeat=self.nb_labels))
        solutions = []
        # it only works for one test item for a moment
        for item in test_dataset:
            maximality_idx = np.ones(len(all_output_space), dtype=bool)
            for i1, m1 in enumerate(all_output_space):
                for i2, m2 in enumerate(all_output_space):
                    if i1 != i2 and maximality_idx[i1] and maximality_idx[i2]:
                        m1_cost_vector, m2_cost_vector = [], []
                        for y in all_output_space:
                            m1_cost_vector.append(hamming_distance(y, m1))
                            m2_cost_vector.append(hamming_distance(y, m2))
                        l_cost_vector = np.array(m2_cost_vector) - np.array(m1_cost_vector)
                        l_exp = self.root.getlowerexpectation(l_cost_vector, item, ncc_s_param, ncc_epsilon)
                        self._logger.debug("%s >_M %s  <=>  %s >? 0 ", m1, m2, l_exp)
                        if l_exp > 0:
                            maximality_idx[i2] = False
            solution_exact = list(compress(all_output_space, maximality_idx))
            self._logger.debug("set solutions exact inference %s", solution_exact)
            solutions.append(solution_exact)

        return solutions
