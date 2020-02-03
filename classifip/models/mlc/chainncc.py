import abc, math, time
from classifip.representations.voting import Scores
from classifip.evaluation.measures import hamming_distance
import numpy as np
from classifip.representations import binaryTree as bt
from classifip.representations.intervalsProbability import IntervalsProbability
from itertools import permutations, combinations, product
import queue
import copy
from classifip.utils import timeit


class BinaryMultiLabel(bt.BinaryTree):

    #@timeit
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


class MLCNCC(metaclass=abc.ABCMeta):

    def __init__(self):
        self.feature_names = []
        self.label_names = []
        self.feature_values = dict()
        self.feature_count = dict()
        self.label_counts = []
        self.nb_labels = 0
        self.training_size = 0

    def learn(self, learn_data_set, nb_labels=1, seed_random_label=None):
        self.__init__()
        self.nb_labels = nb_labels
        self.training_size = len(learn_data_set.data)

        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]

        # Generation random position for chain label
        if seed_random_label is not None:
            np.random.seed(seed_random_label)
            np.random.shuffle(self.label_names)
        self.feature_values = learn_data_set.attribute_data.copy()
        for label_value in self.label_names:
            label_set_one = learn_data_set.select_col_vals(label_value, ['1'])
            self.label_counts.append(len(label_set_one.data))
            label_set_zero = learn_data_set.select_col_vals(label_value, ['0'])

            for feature in self.feature_names:
                count_vector_one, count_vector_zero = [], []
                feature_index = learn_data_set.attributes.index(feature)
                for feature_value in learn_data_set.attribute_data[feature]:
                    nb_items_one = [row[feature_index] for row in label_set_one.data].count(feature_value)
                    count_vector_one.append(nb_items_one)
                    nb_items_zero = [row[feature_index] for row in label_set_zero.data].count(feature_value)
                    count_vector_zero.append(nb_items_zero)
                self.feature_count[label_value + '|in|' + feature] = count_vector_one
                self.feature_count[label_value + '|out|' + feature] = count_vector_zero

            for label_feature in self.label_names:
                if label_feature != label_value:
                    label_feature_index = learn_data_set.attributes.index(label_feature)
                    count_vector_one, count_vector_zero = [], []
                    for label_feature_value in learn_data_set.attribute_data[label_feature]:
                        nb_items_one = [row[label_feature_index] for row in label_set_one.data].count(
                            label_feature_value)
                        count_vector_one.append(nb_items_one)
                        nb_items_zero = [row[label_feature_index] for row in label_set_zero.data].count(
                            label_feature_value)
                        count_vector_zero.append(nb_items_zero)
                    self.feature_count[label_value + '|in|' + label_feature] = count_vector_one
                    self.feature_count[label_value + '|out|' + label_feature] = count_vector_zero

    def lower_upper_probability(self, feature, feature_value, ncc_s_param, feature_class_name, ncc_epsilon):

        def __restricting_idm(probability, ncc_epsilon_ip, len_features):
            return (1 - ncc_epsilon) * probability + ncc_epsilon_ip / len_features

        f_val_index = self.feature_values[feature].index(feature_value)  #
        num_items = float(sum(self.feature_count[feature_class_name]))
        n_fi_c = self.feature_count[feature_class_name][f_val_index]  # n(f_i|c)
        # n(f_i|c)/(n(c)+s), lower probability: t(f_1|c)->0, t(c)->1
        p_lower = (n_fi_c / (num_items + ncc_s_param))
        # (n(f_i|c)+s)/(n(c)+s), upper probability: t(f_1|c)->1, t(c)->1
        p_upper = ((n_fi_c + ncc_s_param) / (num_items + ncc_s_param))
        len_fi = len(self.feature_count[feature_class_name])  # |F_i|
        # some regularization with epsilon
        p_lower = __restricting_idm(p_lower, ncc_epsilon, len_fi)
        p_upper = __restricting_idm(p_upper, ncc_epsilon, len_fi)
        return p_lower, p_upper

    def lower_upper_probability_feature(self, j, label_prior, item, ncc_s_param, ncc_epsilon):
        # (n(c)+st(c))/(N+s), with s=0 (i.e. prior probabilities precise, P(Y))
        u_denominator_0 = 1 - label_prior[j]  # \overline P(Yj=0)
        l_denominator_0 = 1 - label_prior[j]  # \underline P(Yj=0)
        u_numerator_1 = label_prior[j]  # \overline P(Yj=1)
        l_numerator_1 = label_prior[j]  # \underline P(Yj=1)
        for f_index, feature in enumerate(self.feature_names):
            # computation of denominator (label=1)
            feature_class_name = self.label_names[j] + '|in|' + feature  # (f_i, c=1)
            p_lower, p_upper = self.lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                            feature_class_name, ncc_epsilon)
            l_numerator_1 = l_numerator_1 * p_lower  # prod \underline{P}(f_i|c=1)
            u_numerator_1 = u_numerator_1 * p_upper  # prod \overline{P}(f_i|c=1)

            # computation of numerator (label=0)
            feature_class_name = self.label_names[j] + '|out|' + feature
            p_lower, p_upper = self.lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                            feature_class_name, ncc_epsilon)
            l_denominator_0 = l_denominator_0 * p_lower  # prod \underline{P}(f_i|c=0)
            u_denominator_0 = u_denominator_0 * p_upper  # prod \overline{P}(f_i|c=0)

        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0

    def lower_upper_probability_labels(self, j, augmented_labels, ncc_s_param, ncc_epsilon):
        u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = 1, 1, 1, 1
        for l_index, label in enumerate(self.label_names[:len(augmented_labels)]):
            for label_predicted_value in augmented_labels[l_index]:
                # computation of denominator (label=1)
                label_class_name = self.label_names[j] + '|in|' + label  # (l_i=1, c=1)
                p_lower, p_upper = self.lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                label_class_name, ncc_epsilon)
                l_numerator_1 = l_numerator_1 * p_lower  # prod \underline{P}(f_i|c=1)
                u_numerator_1 = u_numerator_1 * p_upper  # prod \overline{P}(f_i|c=1)

                # computation of numerator (label=0)
                label_class_name = self.label_names[j] + '|out|' + label  # (l_i=0, c=0)
                p_lower, p_upper = self.lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                label_class_name, ncc_epsilon)
                l_denominator_0 = l_denominator_0 * p_lower  # prod \underline{P}(f_i|c=0)
                u_denominator_0 = u_denominator_0 * p_upper  # prod \overline{P}(f_i|c=0)
        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0

    def lower_upper_cond_probability(self, j, label_prior, item, chain_predict_labels, ncc_s_param, ncc_epsilon):
        u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
            self.lower_upper_probability_feature(j, label_prior, item, ncc_s_param, ncc_epsilon)

        u_numerator_label_1, l_numerator_label_1, u_denominator_label_0, l_denominator_label_0 = \
            self.lower_upper_probability_labels(j, chain_predict_labels, ncc_s_param, ncc_epsilon)

        u_numerator_1 = u_numerator_1 * u_numerator_label_1
        l_numerator_1 = l_numerator_1 * l_numerator_label_1
        u_denominator_0 = u_denominator_0 * u_denominator_label_0
        l_denominator_0 = l_denominator_0 * l_denominator_label_0
        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0

    def create_partial_vector(self, solutions):
        """
        Naive version calculate partial vector from a set of solutions.
        Here, we consider that the value of a partial solution inside each solution of solutions is 3.
        :return:
        """
        if len(solutions) == 0:
            partial_solution = 4 * np.ones(self.nb_labels, dtype=int)
        else:
            partial_solution = solutions[0]
            for solution in solutions[1:]:
                for j in range(self.nb_labels):
                    if partial_solution[j] == 3 and solution[j] != 3:
                        partial_solution[j] = solution[j]
                    if abs(partial_solution[j] - solution[j]) == 1:
                        partial_solution[j] = 4  # represent partial vector {0,1}
        return [[0, 1] if y == 4 else y for y in partial_solution]


class MLCNCCOuterApprox(MLCNCC, metaclass=abc.ABCMeta):

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2):
        label_prior = [n / float(self.training_size) for n in self.label_counts]
        interval_prob_answers, predict_chain_answers = [], []

        for item in test_dataset:
            # initializing scores
            resulting_score = np.zeros((self.nb_labels, 2))
            chain_predict_labels = []
            # computes lower/upper prob for a chain predict labels
            for j in range(self.nb_labels):
                u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                    super(MLCNCCOuterApprox, self).lower_upper_cond_probability(j, label_prior, item,
                                                                                chain_predict_labels,
                                                                                ncc_s_param,
                                                                                ncc_epsilon)

                # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
                resulting_score[j, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
                resulting_score[j, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)

                if resulting_score[j, 0] > 0.5:
                    chain_predict_labels.append(["1"])
                elif resulting_score[j, 1] < 0.5:
                    chain_predict_labels.append(["0"])
                else:
                    chain_predict_labels.append(["0", "1"])

            result = Scores(resulting_score)
            interval_prob_answers.append(result)
            predict_chain_answers.append(chain_predict_labels)

        return interval_prob_answers, predict_chain_answers


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

        #@timeit
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
