import pandas as pd
import feather
import os
from os.path import join

current_dir = os.getcwd()
out_path = join(current_dir, "../../resources/")


def import_data_set(in_path, sep="\s", name=None, header=None):
    data = pd.read_csv(in_path, sep=sep, header=header)
    if name is None:
        name = os.path.basename(in_path)
        name = os.path.splitext(name)
        name = name[0] + ".data"
    print(data)
    feather.write_dataframe(data, join(out_path, name))


def export_data_set(name = None):
    if name is None:
        data = feather.read_dataframe(join(out_path, "iris.data"))
    else:
        data = feather.read_dataframe(join(out_path, name))
    return data

# root = "~/Dropbox/PhD/testing/data_lda.csv"
# import_data_set(root, sep=",", name="bin_normal_rnd.data", header=0)
# print(export_data_set("bin_normal_rnd.data"))