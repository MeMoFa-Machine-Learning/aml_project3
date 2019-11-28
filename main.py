import pandas as pd
import os.path as ospath
import argparse
import numpy as np
from os import makedirs
from itertools import tee
from csv import reader
import biosppy.signals.ecg as ecg
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from statistics import median as pymedian


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


def extract_rr_interval(rpeaks):
    sum_of_diff = 0
    pairs = list(pairwise(rpeaks))
    rr_intervals = []
    for pair in pairs:
        diff = pair[1] - pair[0]
        sum_of_diff += diff
        rr_intervals.append(diff)
    average_rr_interval = sum_of_diff / len(pairs)
    rr_intervals = sorted(rr_intervals)
    median_rr_interval = pymedian(rr_intervals)
    q1, q3 = np.percentile(rr_intervals, [25, 75])
    iqr = q3 - q1
    return average_rr_interval, median_rr_interval, iqr


def average_r_amplitude(filtered, rpeaks):
    return np.mean(filtered[rpeaks])


def std_r_amplitude(filtered, rpeaks):
    return np.std(filtered[rpeaks])


def median_r_amplitude(filtered, rpeaks):
    return np.median(filtered[rpeaks])


def iqr_r_amplitude(filtered, rpeaks):
    q1, q3 = np.percentile(filtered[rpeaks], [25, 75])
    iqr = q3 - q1
    return iqr


def ecg_domain(mean_template):
    return np.max(mean_template) - np.min(mean_template)


def extract_r_peak(mean_template):
    interval_max_i = np.argmax(mean_template[30:90])
    return np.max(mean_template[30:90]), interval_max_i + 30


def extract_p_peak(mean_template):
    return np.max(mean_template[:35]), np.argmax(mean_template[:35])


def extract_t_peak(mean_template):
    return np.max(mean_template[100:]), np.argmax(mean_template[100:])


def std_t_peak(templates):
    return np.mean(np.std(templates[:, 100:], axis=0))


def std_p_peak(templates):
    return np.mean(np.std(templates[:, :30], axis=0))


def extract_manual_features(samples):
    manual_features_array = []
    for i, raw_ecg in enumerate(tqdm(samples)):
        ts, filtered, rpeaks, templates_ts, templates, heartrates_ts, heartrates = ecg.ecg(raw_ecg, sampling_rate=300,
                                                                                           show=False)
        mean_template = np.mean(templates, axis=0)

        feature_extracted_samples = []
        rr_interval_statistics = extract_rr_interval(rpeaks)
        feature_extracted_samples.append(rr_interval_statistics[0])  # average RR interval
        feature_extracted_samples.append(rr_interval_statistics[1])  # median RR interval
        feature_extracted_samples.append(rr_interval_statistics[2])  # IQR RR interval
        feature_extracted_samples.append(average_r_amplitude(filtered, rpeaks) - median_r_amplitude(filtered, rpeaks))
        feature_extracted_samples.append(std_r_amplitude(filtered, rpeaks))  # standard deviation R amplitude
        feature_extracted_samples.append(iqr_r_amplitude(filtered, rpeaks))  # IQR R amplitude
        feature_extracted_samples.append(ecg_domain(mean_template))
        feature_extracted_samples.append(std_p_peak(templates))
        feature_extracted_samples.append(std_t_peak(templates))
        # average peak amplitudes and indices
        p_peak = extract_p_peak(mean_template)
        t_peak = extract_t_peak(mean_template)
        r_peak = extract_r_peak(mean_template)
        feature_extracted_samples.append(p_peak[0])  # average amplitude of P peak
        feature_extracted_samples.append(t_peak[0])  # average amplitude of T peak
        feature_extracted_samples.append(r_peak[1] - p_peak[1])  # average PR interval
        feature_extracted_samples.append(t_peak[1] - r_peak[1])  # average RT interval
        # slope of P peak: a1
        feature_extracted_samples.append((p_peak[0] - mean_template[p_peak[1] - 2]) / (p_peak[1] - (p_peak[1] - 2)))
        # slope of P peak: a2
        feature_extracted_samples.append((p_peak[0] - mean_template[p_peak[1] + 2]) / (p_peak[1] - (p_peak[1] + 2)))
        # slope of R peak: a3
        feature_extracted_samples.append((r_peak[0] - mean_template[r_peak[1] - 2]) / (r_peak[1] - (r_peak[1] - 2)))
        # slope of R peak: a4
        feature_extracted_samples.append((r_peak[0] - mean_template[r_peak[1] + 2]) / (r_peak[1] - (r_peak[1] + 2)))
        # slope of T peak: a5
        feature_extracted_samples.append((t_peak[0] - mean_template[t_peak[1] - 2]) / (t_peak[1] - (t_peak[1] - 2)))
        # slope of T peak: a6
        feature_extracted_samples.append((t_peak[0] - mean_template[t_peak[1] + 2]) / (t_peak[1] - (t_peak[1] + 2)))

        manual_features_array.append(feature_extracted_samples)
    return np.array(manual_features_array)


def calculate_pre_features(x_pre, y_pre, x_train, x_test):
    classifiers = [
        RandomForestClassifier(class_weight='balanced'),
        KNeighborsClassifier(),
        SVC(class_weight="balanced"),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
    ]

    x_train_pres, x_test_pres = ([], [])
    for classifier in classifiers:
        classifier.fit(x_pre, y_pre)
        x_train_pres.append(classifier.predict(x_train))
        x_test_pres.append(classifier.predict(x_test))

    return np.array(x_train_pres).transpose(), np.array(x_test_pres).transpose()


def main(debug=False, outfile="out.csv"):
    output_pathname = "output"
    output_filepath = ospath.join(output_pathname, outfile)
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

    # Preprocessing Step for meta-feature calculation: StandardScaler
    x_train_fsel, x_test_fsel = perform_data_scaling(x_train_fsel, x_test_fsel)

    # Class-features calculation
    x_train_rest, x_feature, y_train_rest, y_feature = train_test_split(x_train_fsel, y_train_orig, test_size=0.1, random_state=0)

    class_features_train, class_features_test = calculate_pre_features(x_feature, y_feature, x_train_rest, x_test_fsel)
    x_train_rest = np.hstack((x_train_rest, class_features_train))
    x_test_fsel = np.hstack((x_test_fsel, class_features_test))

    # Grid Search
    max_depth         = [3] if debug else [3, 5, 7, 9, 11]
    min_samples_split = [5] if debug else [2, 3, 4, 5, 6, 7]
    n_estimators      = [6] if debug else [50, 200, 350, 500]

    adaboost_tree_selection = [
        DTC(max_depth=1, class_weight='balanced'),
        DTC(max_depth=3, class_weight='balanced'),
        DTC(max_depth=5, class_weight='balanced'),
        DTC(max_depth=7, class_weight='balanced'),
        DTC(max_depth=9, class_weight='balanced'),
        DTC(max_depth=11, class_weight='balanced')
    ]

    ada_base_estimators = [DTC(max_depth=3)] if debug else adaboost_tree_selection

    k_best_features = [x_train_rest.shape[1]] if debug else list(np.linspace(start=5, stop=x_train_rest.shape[1], num=5, endpoint=True, dtype=int))

    models = [
        {
            'model': RandomForestClassifier,
            'parameters': {
                'fs__k': k_best_features,
                'cm__criterion': ['entropy', 'gini'],
                'cm__max_depth': max_depth,
                'cm__min_samples_split': min_samples_split,
                'cm__n_estimators': n_estimators,
                'cm__class_weight': ['balanced'],
            }
        },
        {
            'model': AdaBoostClassifier,
            'parameters': {
                'fs__k': k_best_features,
                'cm__n_estimators': n_estimators,
                'cm__base_estimator': ada_base_estimators,
            }
        }
    ]

    # Perform the cross-validation
    x_train_gs, x_ho, y_train_gs, y_ho = train_test_split(x_train_rest, y_train_rest, test_size=0.1, random_state=0)

    best_models = []
    for model in models:

        pl = Pipeline([('fs', SelectKBest()), ('cm', model['model']())], memory=".")
        kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=6)

        # C-support vector classification according to a one-vs-one scheme
        grid_search = GridSearchCV(pl, model['parameters'], scoring="f1_micro", n_jobs=-1, cv=kfold, verbose=1)
        grid_result = grid_search.fit(x_train_gs, y_train_gs)
        # Calculate statistics and calculate on hold-out
        logging.info(
            "Best for model %s: %f using %s" % (str(model['model']), grid_result.best_score_, grid_result.best_params_))
        y_ho_pred = grid_search.predict(x_ho)
        hold_out_score = f1_score(y_ho_pred, y_ho, average='micro')
        best_models.append((hold_out_score, grid_result.best_params_, model['model']))
        logging.info("Best score on hold-out: {}".format(hold_out_score))

    # Pick best params
    final_model_params_i = int(np.argmax(np.array(best_models)[:, 0]))
    final_model_type = best_models[final_model_params_i][2]
    final_model_params = best_models[final_model_params_i][1]
    logging.info("Picked the following model {} with params: {}".format(str(final_model_type), final_model_params))

    # Fit final model
    logging.info("Fitting the final model...")
    final_model = Pipeline([('fs', SelectKBest()), ('cm', final_model_type())])
    final_model.set_params(**final_model_params)
    final_model.fit(x_train_gs, y_train_gs)

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
    parser.add_argument("--outfile", required=False, default="out.csv")
    args = parser.parse_args()

    main(debug=args.debug, outfile=args.outfile)
