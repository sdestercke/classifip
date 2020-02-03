from classifip.evaluation.measures import hamming_distance
from classifip.representations import binaryTree as bt
from classifip.representations.intervalsProbability import IntervalsProbability
from itertools import permutations, combinations, product
import queue, copy, math, abc, time
from classifip.utils import timeit
import numpy as np
from .mlcncc import MLCNCC


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

    def __init__(self):
        super(MLCNCCExact, self).__init__()
        self.power_set = []
        self.root = None

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

    def evaluate(self, test_dataset, ncc_s_param=2, ncc_epsilon=0.001):
        solutions = []
        indices_labels = list(range(self.nb_labels))
        all_output_space = list(product([0, 1], repeat=self.nb_labels))

        # @timeit
        def calculation_cost(a_sub_vector, neq_idx_labels=None):
            _cost_vector = []
            for y in all_output_space:
                y_sub_array = np.array(y)[neq_idx_labels] if neq_idx_labels is not None else y
                _cost_vector.append(hamming_distance(y_sub_array, a_sub_vector))
            return np.array(_cost_vector)

        # it only works for one test item for a moment
        for item in test_dataset:
            start = time.time()
            for i in range(self.nb_labels - 1):
                n_not_equal_indices = (self.nb_labels - 1) - i
                a_set_indices = list(combinations(indices_labels, n_not_equal_indices))
                # print(n_not_equal_indices, a_set_indices)
                for neq_idx in a_set_indices:
                    for a_vector in product([0, 1], repeat=n_not_equal_indices):
                        cost_vector = calculation_cost(list(a_vector), list(neq_idx))
                        inf_expectation = self.root.getlowerexpectation(cost_vector, item, ncc_s_param, ncc_epsilon)
                        # print(a_vector, n_not_equal_indices, cost_vector, inf_expectation)
                        # print("->", n_not_equal_indices, neq_idx, a_vector, cost_vector, inf_expectation)
                        if (n_not_equal_indices * 0.5) > inf_expectation:
                            no_one_prediction = 3 * np.ones(self.nb_labels, dtype=int)
                            no_one_prediction[list(neq_idx)] = list(a_vector)
                            # print("solution", no_one_prediction, a_vector, inf_expectation, neq_idx, cost_vector)
                            solutions.append(no_one_prediction)

            for a_vector in product([0, 1], repeat=self.nb_labels):
                cost_vector = calculation_cost(list(a_vector))
                # print("->", not_a_vector, cost_vector)
                inf_expectation = self.root.getlowerexpectation(cost_vector, item, ncc_s_param, ncc_epsilon)
                # print(a_vector, cost_vector, inf_expectation)
                if (self.nb_labels * 0.5) > inf_expectation:
                    # print("solution", not_a_vector, a_vector, inf_expectation, cost_vector)
                    solutions.append(np.array(a_vector))
            end = time.time()
            # print("Time-Inference-Instance %s " % (end - start))

        # print(solutions)
        # (0) remove cycle transition
        start = time.time()
        for solution in solutions.copy():
            inverse = []
            for label in solution:
                if label == 3:
                    inverse.append(3)
                else:
                    inverse.append(1 - label)
            for idx, sol2 in enumerate(solutions.copy()):
                if np.array_equal(sol2, inverse):
                    del solutions[idx]
                    break

        # print(solutions)
        # (1) expansion binary vector
        solution_expansion = []
        for solution in solutions:
            if 3 in solution:
                current = np.array([], dtype=int)
                q_current = queue.Queue()
                for label in solution:
                    q_next = queue.Queue()
                    if not q_current.empty():
                        while not q_current.empty():
                            current = q_current.get()
                            if label == 3:
                                q_next.put(np.append(current, 1))
                                q_next.put(np.append(current, 0))
                            else:
                                q_next.put(np.append(current, label))
                    else:
                        if label == 3:
                            q_next.put(np.append(current, 1))
                            q_next.put(np.append(current, 0))
                        else:
                            q_next.put(np.append(current, label))
                    q_current = q_next

                while not q_current.empty():
                    solution_expansion.append(tuple(q_current.get()))
            else:
                solution_expansion.append(tuple(solution))

        solution_dominated = []
        for solution in solutions:
            if 3 in solution:
                current = np.array([], dtype=int)
                q_current = queue.Queue()
                for label in solution:
                    q_next = queue.Queue()
                    if not q_current.empty():
                        while not q_current.empty():
                            current = q_current.get()
                            if label == 3:
                                q_next.put(np.append(current, 1))
                                q_next.put(np.append(current, 0))
                            else:
                                q_next.put(np.append(current, 1 - label))
                    else:
                        if label == 3:
                            q_next.put(np.append(current, 1))
                            q_next.put(np.append(current, 0))
                        else:
                            q_next.put(np.append(current, 1 - label))
                    q_current = q_next

                while not q_current.empty():
                    solution_dominated.append(tuple(q_current.get()))
            else:
                solution_dominated.append(tuple(1 - solution))

        # (2) transitivity application maximality
        solution_exact = []
        # print(solution_expansion)
        # print(solution_dominated)
        for solution in solution_expansion:
            if solution not in solution_dominated:
                solution_exact.append(np.array(solution))
        # print("Time-Expansion-Transition-Cycle %s " % (time.time() - start))

        return self.create_partial_vector(solution_exact)

    def evaluate_exact(self, test_dataset, ncc_s_param=2, ncc_epsilon=0.001):
        solution_exact_set_m2 = set()
        solution_exact_set_m1 = set()
        all_output_space = list(product([0, 1], repeat=self.nb_labels))
        # it only works for one test item for a moment
        for item in test_dataset:
            for i1, m1 in enumerate(all_output_space):
                for i2, m2 in enumerate(all_output_space):
                    if i1 != i2:
                        m1_cost_vector, m2_cost_vector = [], []
                        for y in all_output_space:
                            m1_cost_vector.append(hamming_distance(y, m1))
                            m2_cost_vector.append(hamming_distance(y, m2))
                        l_cost_vector = np.array(m2_cost_vector) - np.array(m1_cost_vector)
                        l_exp = self.root.getlowerexpectation(l_cost_vector, item, ncc_s_param, ncc_epsilon)
                        # print(m1, ">", m2, l_exp)
                        if l_exp > 0:
                            solution_exact_set_m1.add(m1)
                            solution_exact_set_m2.add(m2)

        # verifier transitive maximality solutions (and remove elements dominated)
        solution_exact = []
        for solution in solution_exact_set_m1:
            if solution not in solution_exact_set_m2:
                solution_exact.append(np.array(solution))

        return self.create_partial_vector(solution_exact)
