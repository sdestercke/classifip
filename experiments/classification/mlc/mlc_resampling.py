from mlc_common import init_dataset
from classifip.evaluation import train_test_split
import random, numpy as np


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def generate_name(name, resampling, pct_test):
    test_full_name = "%s_test_%s_%s.arff" % (name, resampling, pct_test)
    train_full_name = "%s_train_%s_%s.arff" % (name, resampling, pct_test)
    return test_full_name, train_full_name


def saving_data_sets(data_learning, out_path, pct_test, file_train_name, file_test_name, seed):
    data_training, data_test = \
        train_test_split(data_learning, test_pct=pct_test, random_seed=seed)
    # testing data set saving
    data_test.comment = "Testing re-sampling generation at " + str(pct_test) + "%."
    data_test.comment += "With seed " + str(seed) + "."
    f_write = open(out_path + file_test_name, "w")
    f_write.write(data_test.write())
    f_write.close()
    # training data set saving
    data_training.comment = "Training re-sampling generation at " + str(1 - pct_test) + "%."
    data_training.comment += "With seed " + str(seed) + "."
    f_write = open(out_path + file_train_name, "w")
    f_write.write(data_training.write())
    f_write.close()


def create_dataset_by_percentage(in_path, out_path, pct_test, nb_samplings):
    seeds = generate_seeds(nb_samplings)
    data_learning, _ = init_dataset(in_path, remove_features=[], scaling=True)
    print("[SEEDs-GENERATION-PCT-TEST] ", pct_test, seeds, flush=True)
    for i in range(nb_samplings):
        str_pct_training = str(int(round((1 - pct_test) * 100)))
        file_test_name, file_train_name = generate_name("emotions", str(i + 1), str_pct_training)
        saving_data_sets(data_learning, out_path, pct_test, file_train_name, file_test_name, seeds[i])


in_path = "/Users/salmuz/Downloads/datasets_mlc/emotions.arff"
out_path = "/Users/salmuz/Downloads/"
for pct in np.arange(0.1, 1, 0.1):
    create_dataset_by_percentage(in_path, out_path, pct, 50)
