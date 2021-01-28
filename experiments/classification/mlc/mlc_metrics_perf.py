from mlc_common import *
import numpy as np


class MetricsPerformances:

    def __init__(self,
                 do_inference_exact=False,
                 epsilon_rejects=None,
                 list_constants_spe=None,
                 list_constants_par=None):
        # parameters
        self.epsilon_rejects = epsilon_rejects
        self.list_constants_spe = list_constants_spe
        self.list_constants_par = list_constants_par
        # measures performances
        self.do_inference_exact = do_inference_exact
        self.ich_exact_skeptic, self.cph_exact_skeptic = dict(), dict()
        self.ich_iid_skeptic, self.cph_iid_skeptic, = dict(), dict()
        self.jacc_exact_skeptic, self.score_hamming = dict(), dict()
        # measures of reject option
        self.ich_reject, self.cph_reject, self.jacc_reject = dict(), dict(), dict()
        # measures of abstained option
        self.ich_spe_partial, self.cph_spe_partial = dict(), dict()
        self.ich_par_partial, self.cph_par_partial = dict(), dict()
        self.spe_partial_score, self.par_partial_score = dict(), dict()

    def init_sub_level(self, sub_level):
        """
            Sub level normally for discretization about NCC problem
        :param sub_level:
        :return:
        """
        # exact inference metrics
        self.ich_exact_skeptic[sub_level], self.cph_exact_skeptic[sub_level] = dict(), dict()
        self.jacc_exact_skeptic[sub_level] = dict()
        # binary relevance metrics
        self.ich_iid_skeptic[sub_level], self.cph_iid_skeptic[sub_level] = dict(), dict()
        self.score_hamming[sub_level] = dict()
        if self.epsilon_rejects is not None:
            self.ich_reject[sub_level], self.cph_reject[sub_level] = dict(), dict()
            self.jacc_reject[sub_level] = dict()

        if self.list_constants_spe is not None:
            self.ich_spe_partial[sub_level], self.cph_spe_partial[sub_level] = dict(), dict()
            self.spe_partial_score[sub_level] = dict()

        if self.list_constants_par is not None:
            self.ich_par_partial[sub_level], self.cph_par_partial[sub_level] = dict(), dict()
            self.par_partial_score[sub_level] = dict()

    def init_classical_scores(self,
                              param_imprecision,
                              ich_exact_skeptic,
                              cph_exact_skeptic,
                              jacc_exact_skeptic,
                              ich_iid_skeptic,
                              cph_iid_skeptic,
                              score_hamming,
                              ich_reject,
                              cph_reject,
                              jacc_reject):
        ich_exact_skeptic[param_imprecision], cph_exact_skeptic[param_imprecision] = 0, 0
        jacc_exact_skeptic[param_imprecision] = 0
        ich_iid_skeptic[param_imprecision], cph_iid_skeptic[param_imprecision] = 0, 0
        score_hamming[param_imprecision] = 0
        if self.epsilon_rejects is not None:
            ich_reject[param_imprecision] = dict.fromkeys(np.array(self.epsilon_rejects, dtype='<U'), 0)
            cph_reject[param_imprecision] = dict.fromkeys(np.array(self.epsilon_rejects, dtype='<U'), 0)
            jacc_reject[param_imprecision] = dict.fromkeys(np.array(self.epsilon_rejects, dtype='<U'), 0)

    def init_abstention_scores(self,
                               param_imprecision,
                               ich_spe_partial,
                               cph_spe_partial,
                               spe_partial_score,
                               ich_par_partial,
                               cph_par_partial,
                               par_partial_score):
        if self.list_constants_spe is not None:
            ich_spe_partial[param_imprecision] = dict.fromkeys(np.array(self.list_constants_spe, dtype='<U'), 0)
            cph_spe_partial[param_imprecision] = dict.fromkeys(np.array(self.list_constants_spe, dtype='<U'), 0)
            spe_partial_score[param_imprecision] = dict.fromkeys(np.array(self.list_constants_spe, dtype='<U'), (0, 0))

        if self.list_constants_par is not None:
            ich_par_partial[param_imprecision] = dict.fromkeys(np.array(self.list_constants_par, dtype='<U'), 0)
            cph_par_partial[param_imprecision] = dict.fromkeys(np.array(self.list_constants_par, dtype='<U'), 0)
            par_partial_score[param_imprecision] = dict.fromkeys(np.array(self.list_constants_par, dtype='<U'), (0, 0))

    def recovery_sub_level(self, sub_level, *all_metrics):
        if sub_level is None:
            all_metrics = self.ich_exact_skeptic, self.cph_exact_skeptic, self.jacc_exact_skeptic, \
                          self.ich_iid_skeptic, self.cph_iid_skeptic, self.score_hamming, \
                          self.ich_spe_partial, self.cph_spe_partial, self.spe_partial_score, \
                          self.ich_par_partial, self.cph_par_partial, self.par_partial_score, \
                          self.ich_reject, self.cph_reject, self.jacc_reject
        else:
            (ich_exact_skeptic, cph_exact_skeptic, jacc_exact_skeptic, ich_iid_skeptic,
             cph_iid_skeptic, score_hamming, ich_spe_partial, cph_spe_partial,
             spe_partial_score, ich_par_partial, cph_par_partial, par_partial_score,
             ich_reject, cph_reject, jacc_reject) = all_metrics

            all_metrics = (ich_exact_skeptic[sub_level], cph_exact_skeptic[sub_level],
                           jacc_exact_skeptic[sub_level], ich_iid_skeptic[sub_level],
                           cph_iid_skeptic[sub_level], score_hamming[sub_level],
                           ich_spe_partial[sub_level], cph_spe_partial[sub_level],
                           spe_partial_score[sub_level], ich_par_partial[sub_level],
                           cph_par_partial[sub_level], par_partial_score[sub_level],
                           ich_reject[sub_level], cph_reject[sub_level], jacc_reject[sub_level])
        return all_metrics

    def init_level_imprecision(self, param_imprecision, sub_level=None):
        all_metrics = self.recovery_sub_level(sub_level=None)

        (ich_exact_skeptic, cph_exact_skeptic, jacc_exact_skeptic, ich_iid_skeptic,
         cph_iid_skeptic, score_hamming, ich_spe_partial, cph_spe_partial,
         spe_partial_score, ich_par_partial, cph_par_partial, par_partial_score,
         ich_reject, cph_reject, jacc_reject) = self.recovery_sub_level(sub_level, *all_metrics)

        self.init_classical_scores(param_imprecision,
                                   ich_exact_skeptic,
                                   cph_exact_skeptic,
                                   jacc_exact_skeptic,
                                   ich_iid_skeptic,
                                   cph_iid_skeptic,
                                   score_hamming,
                                   ich_reject,
                                   cph_reject,
                                   jacc_reject)
        self.init_abstention_scores(param_imprecision,
                                    ich_spe_partial,
                                    cph_spe_partial,
                                    spe_partial_score,
                                    ich_par_partial,
                                    cph_par_partial,
                                    par_partial_score)

    def compute_metrics_performance(self, y_true, y_skeptical_exact,
                                    y_br_skeptical, y_precise,
                                    y_eq_1_precise_probs, nb_tests,
                                    param_imprecision, sub_level=None):
        nb_labels = len(y_true)
        all_metrics = self.recovery_sub_level(sub_level=None)
        all_metrics = self.recovery_sub_level(sub_level, *all_metrics)

        (ich_exact_skeptic, cph_exact_skeptic, jacc_exact_skeptic, ich_iid_skeptic,
         cph_iid_skeptic, score_hamming, ich_spe_partial, cph_spe_partial,
         spe_partial_score, ich_par_partial, cph_par_partial, par_partial_score, ich_reject,
         cph_reject, jacc_reject) = all_metrics

        # decompose the partial to full prediction
        y_br_skeptical_full_set = expansion_partial_to_full_set_binary_vector(y_br_skeptical)

        # if enable to do the exact skeptical inference
        inc_ich_skep, inc_cph_skep, inc_jacc = -1, -1, -1
        if self.do_inference_exact:
            y_skeptical_exact_partial = transform_semi_partial_vector(y_skeptical_exact, nb_labels)
            inc_ich_skep, inc_cph_skep = incorrectness_completeness_measure(y_true, y_skeptical_exact_partial)
            inc_jacc = compute_jaccard_similarity_score(y_br_skeptical_full_set, y_skeptical_exact)
        ich_exact_skeptic[param_imprecision] += inc_ich_skep / nb_tests
        cph_exact_skeptic[param_imprecision] += inc_cph_skep / nb_tests
        jacc_exact_skeptic[param_imprecision] += inc_jacc / nb_tests

        # computing imprecise binary relevance metrics
        inc_ich_skep, inc_cph_skep = incorrectness_completeness_measure(y_true, y_br_skeptical)
        inc_acc_prec, _ = incorrectness_completeness_measure(y_true, y_precise)
        ich_iid_skeptic[param_imprecision] += inc_ich_skep / nb_tests
        cph_iid_skeptic[param_imprecision] += inc_cph_skep / nb_tests
        score_hamming[param_imprecision] += inc_acc_prec / nb_tests

        # why incorrectness < accuracy
        if inc_ich_skep > inc_acc_prec:
            print("[inc < acc](outer, precise, ground-truth) ",
                  y_br_skeptical, y_precise, y_true, inc_ich_skep, inc_acc_prec, flush=True)

        # computing abstention metrics
        rs = abstention_partial_hamming_measure(y_true, y_eq_1_precise_probs,
                                                self.list_constants_spe,
                                                self.list_constants_par)
        y_sep_prediction, y_sep_score = rs[0], rs[1]
        y_par_prediction, y_par_score = rs[2], rs[3]
        for c_spe, c_par in zip(self.list_constants_spe, self.list_constants_par):
            rs = incorrectness_completeness_measure(y_true, y_sep_prediction[str(c_spe)])
            ich_spe_partial[param_imprecision][str(c_spe)] += rs[0] / nb_tests
            cph_spe_partial[param_imprecision][str(c_spe)] += rs[1] / nb_tests
            spe_partial_score[param_imprecision][str(c_spe)] = tuple(map(lambda x, y: x + y / nb_tests,
                                                                         spe_partial_score[param_imprecision][str(c_spe)],
                                                                         y_sep_score[str(c_spe)]))
            rs = incorrectness_completeness_measure(y_true, y_par_prediction[str(c_par)])
            ich_par_partial[param_imprecision][str(c_par)] += rs[0] / nb_tests
            cph_par_partial[param_imprecision][str(c_par)] += rs[1] / nb_tests
            par_partial_score[param_imprecision][str(c_par)] = tuple(map(lambda x, y: x + y / nb_tests,
                                                                         par_partial_score[param_imprecision][str(c_par)],
                                                                         y_par_score[str(c_par)]))

        # computing reject metrics
        y_reject_predictions = reject_partial_hamming_measure(self.epsilon_rejects,
                                                              y_eq_1_precise_probs,
                                                              nb_labels)
        for epsilon, y_reject in y_reject_predictions.items():
            inc_ich_reject, inc_cph_reject = incorrectness_completeness_measure(y_true, y_reject)
            ich_reject[param_imprecision][epsilon] += inc_ich_reject / nb_tests
            cph_reject[param_imprecision][epsilon] += inc_cph_reject / nb_tests

            y_reject_full_set = expansion_partial_to_full_set_binary_vector(y_reject)
            inc_jacc_reject = compute_jaccard_similarity_score(y_br_skeptical, y_reject_full_set)
            jacc_reject[param_imprecision][epsilon] += inc_jacc_reject / nb_tests

    def generate_row_line(self, param_imprecision, time, k_fold, sub_level=None):
        all_metrics = self.recovery_sub_level(sub_level=None)
        _partial_saving = list()
        if sub_level is not None:
            _partial_saving.append(str(sub_level))
            all_metrics = self.recovery_sub_level(sub_level, *all_metrics)

        (ich_exact_skeptic, cph_exact_skeptic, jacc_exact_skeptic, ich_iid_skeptic,
         cph_iid_skeptic, score_hamming, ich_spe_partial, cph_spe_partial,
         spe_partial_score, ich_par_partial, cph_par_partial, par_partial_score, ich_reject,
         cph_reject, jacc_reject) = all_metrics

        ich_exact_skeptic[param_imprecision] = ich_exact_skeptic[param_imprecision] / k_fold
        cph_exact_skeptic[param_imprecision] = cph_exact_skeptic[param_imprecision] / k_fold
        ich_iid_skeptic[param_imprecision] = ich_iid_skeptic[param_imprecision] / k_fold
        cph_iid_skeptic[param_imprecision] = cph_iid_skeptic[param_imprecision] / k_fold
        score_hamming[param_imprecision] = score_hamming[param_imprecision] / k_fold
        jacc_exact_skeptic[param_imprecision] = jacc_exact_skeptic[param_imprecision] / k_fold
        _partial_saving.extend([param_imprecision, time,
                                ich_exact_skeptic[param_imprecision],
                                cph_exact_skeptic[param_imprecision],
                                jacc_exact_skeptic[param_imprecision],
                                ich_iid_skeptic[param_imprecision],
                                cph_iid_skeptic[param_imprecision],
                                score_hamming[param_imprecision]])

        _abstention_ich_spe = [e / k_fold for e in ich_spe_partial[param_imprecision].values()]
        _abstention_cph_spe = [e / k_fold for e in cph_spe_partial[param_imprecision].values()]
        _abstention_score_spe, _abstention_nb_abs_spe = list(), list()
        for abs_score, pct_nb_abstentions in spe_partial_score[param_imprecision].values():
            _abstention_score_spe.append(abs_score / k_fold)
            _abstention_nb_abs_spe.append(pct_nb_abstentions / k_fold)
        _partial_saving = _partial_saving + _abstention_ich_spe + _abstention_cph_spe
        _partial_saving = _partial_saving + _abstention_score_spe + _abstention_nb_abs_spe
        _abstention_ich_par = [e / k_fold for e in ich_par_partial[param_imprecision].values()]
        _abstention_cph_par = [e / k_fold for e in cph_par_partial[param_imprecision].values()]
        _abstention_score_par, _abstention_nb_abs_par = list(), list()
        for abs_score, pct_nb_abstentions in par_partial_score[param_imprecision].values():
            _abstention_score_par.append(abs_score / k_fold)
            _abstention_nb_abs_par.append(pct_nb_abstentions / k_fold)
        _partial_saving = _partial_saving + _abstention_ich_par + _abstention_cph_par
        _partial_saving = _partial_saving + _abstention_score_par + _abstention_nb_abs_par

        if self.epsilon_rejects is not None:
            _reject_ich = [e / k_fold for e in ich_reject[param_imprecision].values()]
            _reject_cph = [e / k_fold for e in cph_reject[param_imprecision].values()]
            _reject_jacc = [e / k_fold for e in jacc_reject[param_imprecision].values()]
            _partial_saving = _partial_saving + _reject_ich + _reject_cph + _reject_jacc
        else:
            _reject_ich, _reject_cph, _reject_jacc = [], [], []

        return _partial_saving

    def __str__(self):
        return ""
