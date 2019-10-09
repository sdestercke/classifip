import json
import math
import os
import random
import sys
import time
import inspect

import matplotlib.pyplot as plt
from constraint import *
from enum import Enum


def correctness_distance(_xa, _xb, ya, yb):
    return abs(ya - yb)


def euclidean_distance(xa, xb, ya, yb):
    return math.sqrt(pow(xa - xb, 2) + pow(ya - yb, 2))


class TypeDistance(Enum):
    CORRECTNESS = correctness_distance,
    EUCLIDEAN = euclidean_distance


class TypeMeasure(Enum):
    CORRECTNESS = 1
    COMPLETENESS = 2


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def get_s_values(results_dict, fnt_distance):
    list_s = sorted(list(results_dict.keys()))
    x = [results_dict[s]['comp'] for s in list_s]
    y = [results_dict[s]['corr'] for s in list_s]
    distances = []

    for i in range(len(list_s) - 1):
        distances.append(fnt_distance(x[i], x[i + 1], y[i], y[i + 1]))
    index = distances.index(max(distances))

    new_s = (list_s[index] + list_s[index + 1]) / 2

    return new_s


def _pinfo(message, kwargs):
    current_frame = inspect.currentframe()
    call_frame = inspect.getouterframes(current_frame, 2)
    _fnt_called_name = call_frame[1][3]
    print(time.strftime('%x %X %Z'), _fnt_called_name, "-", message % kwargs, flush=True)


def find_dichotomy(l_s, r_s, type_measure, fnt_get_measures, lower_bnd=0.95, upper_bnd=1, **kwargs_fnt):
    _pinfo("Call finding dichotomy (l_s:%s, r_s:%s, type_measure:%s)", (l_s, r_s, type_measure.name))
    if l_s != r_s:
        mid = round((r_s + l_s) / 2, 2)
    else:
        _pinfo("Not middle, l-value saved %s", l_s)
        l_corr, l_comp = fnt_get_measures(s_current=l_s, **kwargs_fnt)
        return l_s, l_corr, l_comp

    # Calculate mid corr and comp scores
    mid_corr, mid_comp = fnt_get_measures(s_current=mid, **kwargs_fnt)
    mid_measure = mid_corr if type_measure == TypeMeasure.CORRECTNESS else mid_comp
    _pinfo("Find dichotomy (l_s:%s, mid_s:%s, r_s:%s, s_mid_measure:%s, type_measure:%s)",
           (l_s, mid, r_s, mid_measure, type_measure.name))

    if lower_bnd <= mid_measure < upper_bnd:
        _pinfo("Found with dichotomy (mid_s:%s, s_mid_measure:%s, type_measure:%s)",
               (mid, mid_measure, type_measure.name))
        if type_measure == TypeMeasure.CORRECTNESS:
            return mid, mid_measure, mid_comp
        else:
            return mid, mid_corr, mid_measure
    else:
        l_corr, l_comp = fnt_get_measures(s_current=l_s, **kwargs_fnt)
        r_corr, r_comp = fnt_get_measures(s_current=r_s, **kwargs_fnt)
        is_right_bigger = l_corr < r_corr if type_measure == TypeMeasure.CORRECTNESS else l_comp < r_comp
        if is_right_bigger:
            if mid_measure < 1:
                return find_dichotomy(mid, r_s, type_measure, fnt_get_measures, **kwargs_fnt)
            if mid_measure >= 1:
                return find_dichotomy(l_s, mid, type_measure, fnt_get_measures, **kwargs_fnt)
        else:
            if mid_measure < 1:
                return find_dichotomy(l_s, mid, type_measure, fnt_get_measures, **kwargs_fnt)
            if mid_measure >= 1:
                return find_dichotomy(mid, r_s, type_measure, fnt_get_measures, **kwargs_fnt)


def find_min_or_max(s_current, type_measure, fnt_get_measures, kwargs_fnt, lower_bnd=0.95, upper_bnd=1):
    _pinfo("Call finding min-or-max (s_current:%s, type_measure:%s, lower_bnd:%s, upper_bnd:%s)",
           (s_current, type_measure.name, lower_bnd, upper_bnd))
    s_orig, s_prev, l_s, r_s = s_current, 0, 0, 0
    while r_s == 0:
        s_correctness, s_completeness = fnt_get_measures(s_current=s_current, **kwargs_fnt)
        s_measure = s_correctness if type_measure == TypeMeasure.CORRECTNESS else s_completeness
        _pinfo("Find min-or-max (s_current:%s, s_prev:%s, s_measure:%s, type_measure:%s)",
               (s_current, s_prev, s_measure, type_measure.name))

        # checking if measurement is into interval
        is_interval_allowed = (lower_bnd <= s_measure < upper_bnd)
        if type_measure == TypeMeasure.CORRECTNESS:
            is_interval_allowed = (is_interval_allowed and s_orig != s_current)

        if is_interval_allowed:
            _pinfo("Found without dichotomy (s_current:%s, s_measure:%s, type_measure:%s)",
                   (s_current, s_measure, type_measure.name))
            return s_current, s_correctness, s_completeness
        else:
            is_least_one = (s_measure < 1) if type_measure == TypeMeasure.CORRECTNESS \
                else (s_measure >= 1)
            if is_least_one:
                if s_prev == s_current * 2:
                    r_s = s_current * 2
                    l_s = s_current
                else:
                    s_prev = s_current
                    s_current = s_current * 2
            else:
                if s_prev == s_current / 2:
                    l_s = s_current / 2
                    r_s = s_current
                else:
                    s_prev = s_current
                    s_current = s_current / 2
    return find_dichotomy(l_s, r_s, type_measure, fnt_get_measures, **kwargs_fnt,
                          lower_bnd=lower_bnd, upper_bnd=upper_bnd)


def plot_save_results(results_dict, dataset_name, file_name, criterion="correctness", out_root="."):
    folder_name = criterion + '_distance_results/' + dataset_name
    out_relative_path = os.path.join(out_root, folder_name)
    # create folder to save plots
    if not os.path.exists(out_relative_path):
        os.makedirs(out_relative_path)

    # save results into json file
    with open(out_relative_path + "/%s.json" % file_name, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)

    x, y = list(), list()
    impression = list()
    for discretization in results_dict.keys():
        list_s = sorted(list(results_dict[discretization].keys()))
        for s_imprecision in list_s:
            x.append(results_dict[discretization][s_imprecision]['comp'])
            y.append(results_dict[discretization][s_imprecision]['corr'])
            impression.append(str(round(s_imprecision, 2)))

    plt.figure()
    plt.plot(x, y)
    plt.scatter(x, y)

    for text in impression:
        plt.annotate(text, (x[impression.index(text)], y[impression.index(text)]))

    plt.xlabel('Completeness')
    plt.ylabel('Correctness')
    plt.title('Discretization with s values')
    plt.grid(True)
    plt.savefig(out_relative_path + "/%s.png" % file_name)

    plt.close()


def testing_csp():
    problem = Problem()
    problem.addVariable("a", [2])
    problem.addVariable("c", [1])
    problem.addVariable("b", [3])
    problem.addConstraint(AllDifferentConstraint())
    print(problem.getSolutions())
