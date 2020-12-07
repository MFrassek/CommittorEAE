import numpy as np


def measure_correlation(snapshots, correlation_threshold):
    correlated_inputs = get_list_of_correlated_inputs(
        snapshots, correlation_threshold)
    if len(correlated_inputs) > 0:
        print(("Caution!\nCorrelation between input data can affect the "
              + "reliability of the importance measure.\n"
              + "Correlations of more than {} "
              + "were found between {} pair(s) of input variables:\n\t{}\n")
              .format(correlation_threshold,
              len(correlated_inputs),
              "\n\t".join([convert_correlation_list_entry_to_string(entry)
                           for entry in correlated_inputs])))
    else:
        print("No correlation above {} was found between the inputs."
              .format(correlation_threshold))
    return correlated_inputs


def get_list_of_correlated_inputs(snapshots, correlation_threshold):
    return [make_correlation_list_entry(row_nr, col_nr, entry)
            for row_nr, row in enumerate(get_covariance_matrix(snapshots))
            for col_nr, entry in enumerate(row)
            if row_nr > col_nr and abs(entry) >= correlation_threshold]


def get_covariance_matrix(snapshots):
    return np.cov(np.transpose(snapshots))


def make_correlation_list_entry(row_nr, col_nr, entry):
    return [str(row_nr), str(col_nr), "{:.3f}".format(entry)]


def convert_correlation_list_entry_to_string(entry):
    return "{},{}: {}".format(*entry)
