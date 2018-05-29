import pandas as pd
import feather
import os

current_dir = os.getcwd()
out_path = os.path.join(current_dir, "../../resources/")


def import_data_set(in_path, sep="\s", name=None):
		data = pd.read_csv(in_path, sep=sep, header=None)
		if name is None:
				name = os.path.basename(in_path)
				name = os.path.splitext(name)
				name = name[0] + ".data"
		feather.write_dataframe(data, os.path.join(out_path, name))


def export_data_set(name):
		pass


root = "/Users/salmuz/Downloads/iris.txt"
import_data_set(root, sep=",")
