from mlc_common import init_dataset
from classifip.evaluation import train_test_split
import random


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def create_dataset_by_percentage(in_path, out_path, pct_test, nb_samplings):
    seeds = generate_seeds(nb_samplings)
    data_learning, _ = init_dataset(in_path, remove_features=[], scaling=True)
    print("[SEEDs-GENERATION]", seeds, flush=True)
    for i in range(nb_samplings):
        data_training, data_test = \
            train_test_split(data_learning, test_pct=pct_test, random_seed=seeds[i])
        data_test.comment = "Testing re-sampling generation at " + str(pct_test) + "%."
        data_test.comment += "With seed " + str(seeds[i]) + "."
        f_write = open(out_path + "emotions_test_" + str(i + 1) + "_" + str(int(round((1 - pct_test) * 100))) + ".arff", "w")
        f_write.write(data_test.write())
        f_write.close()
        data_training.comment = "Training re-sampling generation at " + str(1 - pct_test) + "%."
        data_training.comment += "With seed " + str(seeds[i]) + "."
        f_write = open(out_path + "emotions_train_" + str(i + 1) + "_" + str(int(round((1 - pct_test) * 100))) + ".arff", "w")
        f_write.write(data_training.write())
        f_write.close()


in_path = "/Users/salmuz/Downloads/datasets_mlc/emotions.arff"
out_path = "/Users/salmuz/Downloads/"
create_dataset_by_percentage(in_path, out_path, 0.9, 25)