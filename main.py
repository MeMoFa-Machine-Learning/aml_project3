import pandas as pd
import os.path as ospath
import argparse
import numpy as np
from os import makedirs


def main():
    output_pathname = "output"
    output_filepath = ospath.join(output_pathname, "out.csv")
    training_data_dir = ospath.join("data", "training")
    testing_data_dir = ospath.join("data", "testing")

    # read the data
    train_data_x = pd.read_csv(ospath.join(training_data_dir, "X_train.csv"), delimiter=",").drop(["id"], axis=1)
    train_data_y = pd.read_csv(ospath.join(training_data_dir, "y_train.csv"), delimiter=",")["y"]

    test_data_x = pd.read_csv(ospath.join(testing_data_dir, "X_test.csv"), delimiter=",")
    test_data_ids = test_data_x["id"]  # Get the ids for processing later on
    test_data_x = test_data_x.drop(["id"], axis=1)

    # separate the data between label and features and convert them into values (required after using pandas)
    x_train_orig = train_data_x.values
    y_train_orig = train_data_y.values
    x_test_orig = test_data_x.values


if __name__ == "__main__":
    main()
