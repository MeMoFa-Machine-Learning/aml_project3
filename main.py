import pandas as pd
import os.path as ospath
import argparse
import numpy as np
from os import makedirs
from itertools import tee
from csv import reader
import biosppy.signals.ecg as ecg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from tqdm import tqdm

import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s')


# Debug parameters
first_n_lines_input = 50


def read_in_irregular_csv(path_to_file, skip_n_lines=1, debug=False):
    file_array = []
    with open(path_to_file, 'r') as csv_file:
        ecg_reader = reader(csv_file, delimiter=',', quotechar='|')
        for row_to_skip in range(skip_n_lines):
            next(ecg_reader)
        for i, row in enumerate(tqdm(ecg_reader)):
            if debug and i == first_n_lines_input:
                break
            file_array.append(np.array(row[1:], dtype=np.int16))
    return file_array


def perform_data_scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def average_r_separation(rpeaks):
    sum_of_diff = 0
    pairs = list(pairwise(rpeaks))
    for pair in pairs:
        sum_of_diff += pair[1] - pair[0]
    return sum_of_diff / len(pairs)


def average_r_amplitude(filtered, rpeaks):
    return np.mean(filtered[rpeaks])


def std_r_amplitude(filtered, rpeaks):
    return np.std(filtered[rpeaks])


def median_r_amplitude(filtered, rpeaks):
    return np.median(filtered[rpeaks])


def ecg_domain(mean_template):
    return np.max(mean_template) - np.min(mean_template)


def extract_manual_features(samples):
    feature_extracted_samples = np.ndarray((len(samples), 4), dtype=np.float64)
    for i, raw_ecg in enumerate(tqdm(samples)):
        ts, filtered, rpeaks, templates_ts, templates, heartrates_ts, heartrates = ecg.ecg(raw_ecg, sampling_rate=300, show=False)
        mean_template = np.mean(templates, axis=0)
        feature_extracted_samples[i][0] = average_r_separation(rpeaks)
        feature_extracted_samples[i][1] = average_r_amplitude(filtered, rpeaks) - median_r_amplitude(filtered, rpeaks)
        feature_extracted_samples[i][2] = std_r_amplitude(filtered, rpeaks)
        feature_extracted_samples[i][3] = ecg_domain(mean_template)
    return feature_extracted_samples


def main(debug=False):
    output_pathname = "output"
    output_filepath = ospath.join(output_pathname, "out.csv")
    training_data_dir = ospath.join("data", "training")
    testing_data_dir = ospath.join("data", "testing")

    # read the data
    logging.info("Reading in training data...")
    train_data_x = read_in_irregular_csv(ospath.join(training_data_dir, "X_train.csv"), debug=debug)
    train_data_y = pd.read_csv(ospath.join(training_data_dir, "y_train.csv"), delimiter=",")["y"]
    if debug:
        train_data_y = train_data_y.head(first_n_lines_input)

    y_train_orig = train_data_y.values
    logging.info("Finished reading in data.")

    # Extract the features of training set
    logging.info("Extracting features...")
    x_train_fsel = extract_manual_features(train_data_x)
    logging.info("Finished extracting features")

    # load raw ECG test data
    logging.info("Reading in testing data...")
    test_data_x = read_in_irregular_csv(ospath.join(testing_data_dir, "X_test.csv"), debug=debug)
    logging.info("Finished reading in data.")

    # Extract the features of testing set
    logging.info("Extracting features...")
    x_test_fsel = extract_manual_features(test_data_x)
    logging.info("Finished extracting features")

    # Preprocessing Step: StandardScaler
    x_train_fsel, x_test_fsel = perform_data_scaling(x_train_fsel, x_test_fsel)

    # Training Step #1: Grid Search
    x_train_gs, x_ho, y_train_gs, y_ho = train_test_split(x_train_fsel, y_train_orig, test_size=0.1, random_state=0)

    reg_param    = [1]       if debug else list(np.logspace(start=-2, stop=2, num=5, endpoint=True, base=10))
    gamma_param  = ['scale'] if debug else list(np.logspace(start=-3, stop=2, num=5, endpoint=True, base=10)) + ['scale']
    degree_param = [2]       if debug else list(np.logspace(start=1, stop=6, num=5, base=1.5, dtype=int))
    max_iters    = [2500]    if debug else [2000, 2500, 3000, ]

    parameters = [
        {
            'svc__kernel': ['rbf'],
            'svc__C': reg_param,
            'svc__gamma': gamma_param,
            'svc__max_iter': max_iters,
            'svc__class_weight': ['balanced']
        },
        {
            'svc__kernel': ['poly'],
            'svc__C': reg_param,
            'svc__gamma': gamma_param,
            'svc__degree': degree_param,
            'svc__max_iter': max_iters,
            'svc__class_weight': ['balanced']
        },
    ]

    # Perform the cross-validation
    best_models = []
    for kernel_params in parameters:

        pl = Pipeline([('svc', SVC())])
        kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=6)

        # C-support vector classification according to a one-vs-one scheme
        grid_search = GridSearchCV(pl, kernel_params, scoring="f1_micro", n_jobs=-1, cv=kfold, verbose=1)
        grid_result = grid_search.fit(x_train_gs, y_train_gs)

        # Calculate statistics and calculate on hold-out
        logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        y_ho_pred = grid_search.predict(x_ho)
        hold_out_score = f1_score(y_ho_pred, y_ho, average='micro')
        best_models.append((hold_out_score, grid_result.best_params_))
        logging.info("Best score on hold-out: {}".format(hold_out_score))

    # Pick best params
    final_model_params_i = int(np.argmax(np.array(best_models)[:, 0]))
    final_model_params = best_models[final_model_params_i][1]
    logging.info("Picked the following model: {}".format(final_model_params))

    # Fit final model
    logging.info("Fitting the final model...")
    final_model = Pipeline([('svc', SVC())])
    final_model.set_params(**final_model_params)
    final_model.fit(x_train_fsel, y_train_orig)

    # Do the prediction
    y_predict = final_model.predict(x_test_fsel)
    unique_elements, counts_elements = np.unique(y_predict, return_counts=True)
    print("test set labels and their corresponding counts")
    print(np.asarray((unique_elements, counts_elements)))

    # Prepare results dataframe
    results = np.zeros((x_test_fsel.shape[0], 2))
    results[:, 0] = list(range(x_test_fsel.shape[0]))
    results[:, 1] = y_predict

    # save the output weights
    if not ospath.exists(output_pathname):
        makedirs(output_pathname)
    np.savetxt(output_filepath, results, fmt=["%1.1f", "%1.1f"], newline="\n", delimiter=",", header="id,y",
               comments="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ECG data")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    main(debug=args.debug)
