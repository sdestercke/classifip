from classifip.evaluation.measures import hamming_distance
from classifip.representations import binaryTree as bt
from classifip.representations.intervalsProbability import IntervalsProbability
from itertools import permutations, combinations, product
import queue, copy, math, abc, time, xxhash
from classifip.utils import timeit
import numpy as np
from .mlcncc import MLCNCC
from classifip.utils import create_logger
from itertools import compress


class BinaryMultiLabel(bt.BinaryTree):

    def getlowerexpectation(self, function, item, ncc_s_param=2, ncc_epsilon=0.001):
        """
           Computing the lower expectation recursively of a multi-label problem,
           by exploring the whole binary tree.
           This procedure works very well for little binary tree with deepness <10.
        :param function:
        :param item:
        :param ncc_s_param:
        :param ncc_epsilon:
        :return:
        """

        class_values = self.node.label
        nb_class = len(class_values)

        if function.shape != (nb_class,):
            raise Exception('Size of cost vector is not correct:', function.shape)
        cost = function

        def lowerExp(NDtree):
            bound_probabilities = NDtree.node.proba(item, ncc_s_param, ncc_epsilon)
            if bound_probabilities.isreachable() == 0:
                # do approximation in precise case with parameter s nearly zero 0
                # check precision 0.9...9 (10 times), supremum should be greater than 1
                sum_sup_prob = bound_probabilities.lproba[0, :].sum()
                if (1 - 1e-10) < sum_sup_prob < 1:
                    idx_big_prob = np.argmin(bound_probabilities.lproba[0, :])
                    value_big_prob = bound_probabilities.lproba[0, idx_big_prob]
                    bound_probabilities.lproba[0, idx_big_prob] = value_big_prob + (1 - sum_sup_prob)
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

    def getlowerexpectationGeneration(self, item, ncc_s_param=2, ncc_epsilon=0.001):
        """
            A improved version of the lower expectation calculation in a multi-label problem.
            This solution recursively walk the binary tree just one time and it builds a Python function
            in string to accelarate quickly the calculation.

            This procedure works very well for little binary tree with deepness < 20
            and with a precision numeric decimal of
                |getlowerexpectationGeneration - getlowerexpectation| < 1e-16

        :param item:
        :param ncc_s_param:
        :param ncc_epsilon:
        :return:
        """
        leafs = dict()
        leafs_var = dict()

        def __hash(query):
            _hash = xxhash.xxh64()
            _hash.update(query)
            res_hash = _hash.hexdigest()
            _hash.reset()
            return res_hash

        def binaryToDecimal(n):
            return int(n, 2)

        def getlowerexpectation(prob, is_leaf=False, idx0=None, idx1=None):
            if is_leaf:
                expDichStr = "min(c[%s]*%s+c[%s]*%s,c[%s]*%s+c[%s]*%s)" % \
                             (idx0, prob[1, 0], idx1, prob[0, 1], idx0, prob[0, 0], idx1, prob[1, 1])
                name_var = "a" + str(idx0) + str(idx1)
                leafs_var[__hash(expDichStr)] = name_var + "=" + expDichStr
                leafs[__hash(expDichStr)] = name_var
            else:
                keyx0 = __hash(idx0)
                new_idx0 = leafs[keyx0] if keyx0 in leafs else idx0
                keyx1 = __hash(idx1)
                new_idx1 = leafs[keyx1] if keyx1 in leafs else idx1
                expDichStr = "min(%s*%s+%s*%s,%s*%s+%s*%s)" \
                             % (new_idx0, prob[1, 0], new_idx1, prob[0, 1], new_idx0, prob[0, 0], new_idx1, prob[1, 1])
                name_var = str(new_idx0) + str(new_idx1)
                leafs_var[__hash(expDichStr)] = name_var + "=" + expDichStr
                leafs[__hash(expDichStr)] = name_var

            return expDichStr

        def lowerExp(NDtree):
            bound_probabilities = NDtree.node.proba(item, ncc_s_param, ncc_epsilon)
            if bound_probabilities.isreachable() == 0:
                # do approximation in precise case with parameter s nearly zero 0
                # check precision 0.9...9 (10 times), supremum should be greater than 1
                sum_sup_prob = bound_probabilities.lproba[0, :].sum()
                if (1 - 1e-10) < sum_sup_prob < 1:
                    idx_big_prob = np.argmin(bound_probabilities.lproba[0, :])
                    value_big_prob = bound_probabilities.lproba[0, idx_big_prob]
                    bound_probabilities.lproba[0, idx_big_prob] = value_big_prob + (1 - sum_sup_prob)
                bound_probabilities.setreachableprobability()

            if NDtree.left.node.count() == 1 and NDtree.left.node.count() == 1:
                expInfStr = getlowerexpectation(bound_probabilities.lproba,
                                                is_leaf=True,
                                                idx0=binaryToDecimal(NDtree.left.node.label[0]),
                                                idx1=binaryToDecimal(NDtree.right.node.label[0]))
            else:
                expInfStr = getlowerexpectation(bound_probabilities.lproba,
                                                idx0=lowerExp(NDtree.left),
                                                idx1=lowerExp(NDtree.right))
            return expInfStr

        lexp = lowerExp(self)
        lexp_cost_var = "\n\t".join(list(leafs_var.values())[:-1])
        return lexp, lexp_cost_var


class MLCNCCExact(MLCNCC):

    def __init__(self, DEBUG=False):
        super(MLCNCCExact, self).__init__(DEBUG)
        self.power_set = []
        self.root = None
        self.DEBUG = DEBUG
        self._logger = create_logger("MLCNCCExact", DEBUG)

    @timeit
    def learn(self,
              learn_data_set,
              nb_labels):
        super(MLCNCCExact, self).learn(learn_data_set, nb_labels)

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
            tree_recursive.node.proba = self.__learning_sub_model(tree_recursive.node.label)
            if len(sub_power_set) > 1:
                tbinary.put(tree_recursive.right)
                tbinary.put(tree_recursive.left)

    def __learning_sub_model(self, labels):
        level_tree_model = int(self.nb_labels - math.log2(len(labels)))
        augmented_input_labels = [str(lab) for lab in labels[0][:level_tree_model]]

        def __inference(item, ncc_s_param=2, ncc_epsilon=0.001):
            u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                super(MLCNCCExact, self).lower_upper_cond_probability(level_tree_model, item,
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
        cardinal_all_output = len(all_output_space)
        all_output_space = np.array(all_output_space)
        PARTIAL_VALUE = MLCNCC.LABEL_PARTIAL_VALUE
        __logger = self._logger

        # creation of binary tree in memory as a function (faster execution instead of recursive method)
        expectation_infimum_str, exp_cost_leaf_var = self.root.getlowerexpectationGeneration(new_instance,
                                                                                             ncc_s_param,
                                                                                             ncc_epsilon)
        exec_function_dynamic = "global exec_expectation_inf\ndef exec_expectation_inf(c):\n\t" + \
                                exp_cost_leaf_var + "\n\treturn " + expectation_infimum_str
        exec(exec_function_dynamic)

        def calculation_cost(a_sub_vector, neq_idx_labels=None):
            """
                Applying by column with a binary singleton Hamming loss function:
                            L(y1, y2) = 1_{y1 \neq y2}
                            L(y1, y2) = 1_{y1 ==0 and y2==1} + 1_{y1 ==1 and y2==0}
                            L(y1, y2) = (1-y1)*y2 + (1-y2)*y1
                            L(y1, y2) = y1 + y2 - 2*y2*y1
                            L(y1, y2) = (y2 - y1)^2 = | y2 - y1 |
                This operation can be computed by column to get vector cost hamming multilabel
            :param a_sub_vector:
            :param neq_idx_labels:
            :return:
            """
            a_fully = np.array([a_sub_vector, ] * cardinal_all_output)
            if neq_idx_labels is not None:
                return np.sum(abs(all_output_space[:, neq_idx_labels] - a_fully), axis=1)
            else:
                return np.sum(abs(all_output_space - a_fully), axis=1)

        def expansion_partial_binary_vector(binary_vector):
            new_partial_vector = list()
            new_net_partial_vector = list()
            for label in binary_vector:
                if label == PARTIAL_VALUE:
                    new_partial_vector.append([1, 0])
                    new_net_partial_vector.append([1, 0])
                else:
                    new_partial_vector.append([label])
                    new_net_partial_vector.append([1 - label])

            set_binary_vector = list(product(*new_partial_vector))
            set_dominated_vector = list(product(*new_net_partial_vector))
            return set_binary_vector, set_dominated_vector

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

        def mark_solution_dominated(set_dominated_preds):
            for dominated in set_dominated_preds:
                maximality_sets[dominated] = False

        def inf_not_equal_labels(nb_labels, p_n_not_equal_indices, p_neq_idx=None):
            for a_vector in product([0, 1], repeat=p_n_not_equal_indices):
                partial_prediction = PARTIAL_VALUE * np.ones(nb_labels, dtype=int)
                partial_prediction[p_neq_idx] = np.array(a_vector)
                is_not_dominated, set_dominated_preds = is_solution_not_dominated(partial_prediction)
                __logger.debug("%s ==> is_not_dominated %s", partial_prediction, is_not_dominated)
                if is_not_dominated:
                    not_a_vector = 1 - np.array(a_vector)
                    cost_vector = calculation_cost(not_a_vector, p_neq_idx)
                    # inf_expectation = root.getlowerexpectation(cost_vector, new_instance, ncc_s_param, ncc_epsilon)
                    inf_expectation = exec_expectation_inf(cost_vector)
                    __logger.debug("%s >_M %s ==> %s <? %s", partial_prediction, not_a_vector,
                                   (p_n_not_equal_indices * 0.5), inf_expectation)
                    if (p_n_not_equal_indices * 0.5) < inf_expectation:
                        mark_solution_dominated(set_dominated_preds)

        # some equal labels in comparison (m1 > m2)
        for i in reversed(range(self.nb_labels - 1)):
            n_not_equal_indices = (self.nb_labels - 1) - i
            a_set_indices = list(combinations(indices_labels, n_not_equal_indices))
            for neq_idx in a_set_indices:
                inf_not_equal_labels(self.nb_labels, n_not_equal_indices, list(neq_idx))
        # none equal labels (all different)
        inf_not_equal_labels(self.nb_labels, self.nb_labels)

        solution_exact = list(filter(lambda k: maximality_sets[k], maximality_sets.keys()))
        self._logger.debug("set solutions improved exact inference %s", solution_exact)
        return solution_exact

    def evaluate(self, test_dataset, ncc_s_param=2, ncc_epsilon=0.001, precision=None):
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
