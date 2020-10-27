from os import listdir
import glob
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
import random
import pickle


def make_train_val_test_from_toy(const):
    return make_train_val_test_split_snapshots_from_snapshots(
        *make_snapshots_from_paths(
            *filter_paths(
                *make_paths_from_toy_data(
                    const),
                const),
            const),
        const)


def make_train_val_test_from_TIS_and_TPS(const):
    return make_train_val_test_split_snapshots_from_snapshots(
        *make_snapshots_from_paths(
            *filter_paths(
                *make_paths_from_TIS_and_TPS_data(
                    const),
                const),
            const),
        const)


def make_train_val_test_split_snapshots_from_snapshots(
        snapshots,
        snapshot_labels,
        snapshot_weights,
        snapshot_origins,
        const):
    train_end = int(len(snapshots) * const.train_ratio)
    val_end = train_end + int(len(snapshots) * const.val_ratio)
    origCounter = Counter(snapshot_origins)
    snapshots, snapshot_labels, snapshot_weights, snapshot_origins = \
        shuffle(
            snapshots, snapshot_labels, snapshot_weights, snapshot_origins,
            random_state=42)
    newCounter = Counter(snapshot_origins[:train_end])
    print("\nFraction of snapshots per interface in the test set:")
    for key in origCounter:
        print("    {}:\t{:.3f}".format(
            key, newCounter[key] / origCounter[key]))
    return np.array([*snapshots[:train_end]]), \
        np.array([*snapshot_labels[:train_end]]), \
        np.array([*snapshot_weights[:train_end]]), \
        np.array([*snapshots[train_end:val_end]]), \
        np.array([*snapshot_labels[train_end:val_end]]), \
        np.array([*snapshot_weights[train_end:val_end]]), \
        np.array([*snapshots[val_end:]]), \
        np.array([*snapshot_labels[val_end:]]), \
        np.array([*snapshot_weights[val_end:]])


def calculate_origin_balanced_weights(snapshot_origins):
    origin_cnt = len(set(snapshot_origins))
    snapshot_len = len(snapshot_origins)
    origin_counter = Counter(snapshot_origins)
    origin_balanced_weights = [snapshot_len /
                               (origin_counter[origin] * origin_cnt)
                               for origin in snapshot_origins]
    return np.array(origin_balanced_weights)


def make_snapshots_from_paths(
        paths, path_labels, path_weights, path_origins, const):
    snapshots = []
    snapshot_labels = []
    snapshot_weights = []
    snapshot_origins = []
    for path, path_label, path_weight, path_origin in zip(
            paths, path_labels, path_weights, path_origins):
        path_part_weight = path_weight / len(path)
        for snapshot_nr, snapshot in enumerate(path):
            # Iterate over all indices within each path and append
            # accordingly the snapshot as well as label,
            # weight and origin.
            snapshots.append(snapshot)
            snapshot_weights.append(path_part_weight)
            snapshot_origins.append(path_origin)
            if path_label == "AA":
                snapshot_labels.append(const.AA_label)
            if path_label == "BB":
                snapshot_labels.append(const.BB_label)
            if path_label == "AB":
                if const.progress:
                    snapshot_labels.append(
                        calculate_progress_label(
                            len(path), snapshot_nr, const))
                else:
                    snapshot_labels.append(const.AB_label)
            if path_label == "BA":
                if const.progress:
                    snapshot_labels.append(
                        const.BB_label - calculate_progress_label(
                            len(path), snapshot_nr, const))
                else:
                    snapshot_labels.append(const.BA_label)
    snapshot_weights = np.array(snapshot_weights) / np.mean(snapshot_weights)
    print("Total mean weights: {}".format(np.mean(snapshot_weights)))
    print("Total sum weights: {}".format(np.sum(snapshot_weights)))
    return np.array(snapshots), \
        np.array(snapshot_labels), \
        np.array(snapshot_weights), \
        np.array(snapshot_origins)


def calculate_progress_label(path_len, snapshot_nr, const):
    """Calculate the progress label in such a way, that the first
    snapshot of the current path is assigned the same label as an
    AA path, the last snapshot is assigned the same label as an BB
    path, and all other snapshot labels are mapped lineraly between
    them.
    """
    return (const.BB_label - const.AA_label) \
        * (snapshot_nr) / (path_len - 1.0) \
        + const.AA_label


def filter_paths(paths, path_labels, path_weights, path_origins, const):
    filtered_tuple_list = [(path, label, weight, origin)
                           for (path, label, weight, origin)
                           in zip(
                                paths, path_labels, path_weights, path_origins)
                           if label in const.keep_labels]
    return list(map(list, zip(*filtered_tuple_list)))


def make_paths_from_toy_data(const):
    paths = np.array(
        pickle.load(open("{}/paths.p".format(const.toy_folder_name), "rb")))
    labels = np.array(
        pickle.load(open("{}/labels.p".format(const.toy_folder_name), "rb")))
    weights = np.ones(len(labels))
    origins = np.ones(len(labels))
    paths, labels = \
        shuffle(paths, labels, random_state=42)
    frac_len = int(len(origins) * const.used_toy_frac)
    return paths[:frac_len],\
        labels[:frac_len],\
        weights[:frac_len],\
        origins[:frac_len]


def make_paths_from_TIS_and_TPS_data(const):
    print("Read TIS files")
    TIS_paths, TIS_labels, TIS_weights, TIS_origins = \
        make_paths_from_TIS_data(const)
    print("Read TPS files")
    # Read in the TPS files and generate paths, labels, weights and origins.
    # Weights are chosen based on the minimal weight assigned to the TIS paths.
    TPS_paths, TPS_labels, TPS_weights, TPS_origins = \
        make_paths_from_TPS_data(min(TIS_weights), const)
    # Update the weight of paths at the highest interface and of the TPS paths
    # to correct for the additional paths available.
    TIS_weights, TPS_weights = \
        correct_highest_interface(
            const.TIS_highest_interface_name, TIS_origins,
            TIS_weights, TPS_weights)
    # Return the merges  TIS and TPS arrays
    return np.append(TIS_paths, TPS_paths, axis=0), \
        np.append(TIS_labels, TPS_labels, axis=0), \
        np.append(TIS_weights, TPS_weights, axis=0), \
        np.append(TIS_origins, TPS_origins, axis=0)


def make_paths_from_TIS_data(const):
    folder_name = const.TIS_folder_name
    paths = []
    labels = []
    mc_weights = []
    reweights = []
    origins = []
    for interface_name in listdir(folder_name):
        print(interface_name)
        origin_lines = get_lines_from_file(
            folder_name, interface_name, "path_name*")
        weight_lines = get_lines_from_file(
            folder_name, interface_name, "path_weight*")
        reweight_lines = get_lines_from_file(
            folder_name, interface_name, "rpe_weigh*")
        origin_lines, weight_lines, reweight_lines = \
            shuffle(
                origin_lines, weight_lines, reweight_lines, random_state=42)
        frac_len = int(len(origin_lines) * const.used_TIS_frac)
        print("Total paths: {}\t Used paths: {}"
              .format(len(origin_lines), frac_len))
        origin_names = get_names_from_lines(origin_lines, frac_len, str)
        weight_names = get_names_from_lines(weight_lines, frac_len, int)
        reweight_names = get_names_from_lines(reweight_lines, frac_len, float)
        for file_nr in range(frac_len):
            path = read_path_from_file(
                "{}/{}/light_data/{}".format(
                    folder_name, interface_name, origin_names[file_nr]),
                const.precision)
            if path_outside_state_definition(path, const):
                handle_path_outside_state_definition(
                    origin_names[file_nr])
            else:
                paths.append(path)
                labels.append(determine_label(path, const))
                mc_weights.append(weight_names[file_nr])
                reweights.append(reweight_names[file_nr])
                origins.append(str(interface_name))

    # Multiply the two weight lists.
    weights = np.array(mc_weights) * np.array(reweights)
    return np.array(paths), np.array(labels), \
        np.array(weights), np.array(origins)


def get_lines_from_file(folder_name, interface_name, file_string):
    """Open file with the path "folder_name/interface_name/filestring"
    and return its contents.
    """
    with open(glob.glob("{}/{}/{}".format(
            folder_name, interface_name, file_string))[0], "r") as file:
        lines = file.readlines()
        return lines


def get_names_from_lines(lines, frac_len, function):
    """Take list of lines read from a file, keep the first fract_len
    elements, remove the end of line character at the end of each
    element and convert it to the type definded by function.
    """
    names = list(map(lambda x: function(x[:-1]), lines[:frac_len]))
    return names


def make_paths_from_TPS_data(TPS_weight, const):
    folder_name = const.TPS_folder_name
    paths = []
    labels = []
    origins = []
    for file_name in listdir(folder_name):
        path = read_path_from_file(
            "{}/{}".format(folder_name, file_name),
            const.precision)
        if path_outside_state_definition(path, const):
            handle_path_outside_state_definition(file_name, path)
        else:
            paths.append(path)
            labels.append(determine_label(path, const))
            origins.append("TPS")

    frac_len = int(len(paths) * const.used_TPS_frac)
    print("Total paths: {}\t Used paths: {}".format(len(paths), frac_len))
    weights = [TPS_weight for i in range(frac_len)]
    paths, labels, origins = shuffle(paths, labels, origins, random_state=42)
    return np.array(paths)[:frac_len], np.array(labels)[:frac_len], \
        np.array(weights), np.array(origins)[:frac_len]


def read_path_from_file(file_path, precision):
    """Iterate over all snapshots in the trajectory, remove the
    linebreak character at the end ('\n'), split them along all
    occurences of ' ', drop the first column (snapshot_index)
    transform the strings into floats, and round to the given precision.
    """
    with open(file_path, "r") as file:
        path = file.readlines()
        if path[0].startswith("#"):
            path = path[1:]
        path = np.array(
            [list(map(lambda x: round(float(x), precision),
             snap[:-1].split(" ")[1:])) for snap in path])
        return path


def path_outside_state_definition(path, const):
    return (snapshots_outside_state_definition(path[0], const)
            or snapshots_outside_state_definition(path[-1], const))


def snapshots_outside_state_definition(snapshot, const):
    return snapshot[0] > const.mcg_A and snapshot[0] < const.mcg_B


def handle_path_outside_state_definition(file_name, path):
    print(("Path in {} begins (mcg = {}) or ends"
           + "(mcg = {}) outside of state definition.")
          .format(file_name, path[0][0], path[-1][0]))


def determine_label(path, const):
    if path[0][0] <= const.mcg_A:
        if path[-1][0] <= const.mcg_A:
            return "AA"
        elif path[-1][0] >= const.mcg_B:
            if np.amax(path, axis=0)[8] \
                    >= const.big_C:
                return "AB"
            else:
                return "AC"
    elif path[0][0] >= const.mcg_B:
        if path[-1][0] <= const.mcg_A:
            return "BA"
        elif path[-1][0] >= const.mcg_B:
            return "BB"
    else:
        return "NN"


def correct_highest_interface(
        highest_interface,
        TIS_origins,
        TIS_weights,
        TPS_weights):
    TIS_highest_interface_cnt = len([1 for origin in TIS_origins
                                     if origin == highest_interface])
    TPS_highest_interface_cnt = len(TPS_weights)
    # Determine by which factor the weights at the highest interface
    # need to be corrected.
    update_factor = TIS_highest_interface_cnt \
        / (TIS_highest_interface_cnt + TPS_highest_interface_cnt)
    # Make an array for broadcasting where the positions of the
    # highest interface are indicated with True.
    TIS_highest_interface_mask = TIS_origins == highest_interface
    # Make an array for broadcasting where the positions of the
    # highest interface are indicated with False.
    TIS_highest_interface_antimask = TIS_origins != highest_interface
    # Make an array with the update_factor at all positions of the
    # the highest interface and 0 everywhere else.
    TIS_update_mask = TIS_highest_interface_mask * update_factor
    # Make an array with the update_factor at all positions of the
    # the highest interface and 0 everywhere else.
    TIS_not_update_mask = TIS_highest_interface_antimask * 1
    # Make a full mask for updating the weights of the highest interface
    # without losing the other weights.
    TIS_full_mask = TIS_update_mask + TIS_not_update_mask
    # Update the TIS and TPS weights.
    TIS_weights = TIS_weights * TIS_full_mask
    TPS_weights = TPS_weights * update_factor
    return TIS_weights, TPS_weights


def read_shooting_points(filename):
    print("Read shooting point file")
    with open(filename, "r") as file:
        file.readline()
        shooting_points = file.readlines()
        precision = 2
        shooting_points = np.array(
            [list(map(lambda x: round(float(x), precision),
             point[:-1].split(" ")[1:])) for point in shooting_points])
        labels = np.array([point[0] for point in shooting_points])
        shooting_points = np.array([point[1:] for point in shooting_points])
        print("{} shooting points read".format(len(labels)))
        return shooting_points, labels


def get_toy_paths(const):
    paths = []
    labels = const.keep_labels
    for label in labels:
        paths.append(get_one_toy_path(const.toy_folder_name, label))
    return paths, labels


def get_TPS_and_TIS_paths(const):
    paths = []
    TIS_labels = sorted(sorted(listdir(const.TIS_folder_name)), key=len)
    for label in TIS_labels:
        paths.append(get_one_TIS_path(const=const, interface=label))
    paths.append(get_one_TPS_path(const=const))
    labels = [convert_interface_name_to_math_text(TIS_interface_name)
              for TIS_interface_name in TIS_labels] + ["$TPS$"]
    return paths, labels


def get_one_TIS_path(const, interface, seed=42):
    path = read_path_from_file(
        get_file_path_with_randomly_chosen_file(
            "{}/{}/light_data".format(const.TIS_folder_name, interface), seed),
        2)
    print("Label: {}".format(determine_label(path, const)))
    return path


def get_one_TPS_path(const, seed=42):
    path = read_path_from_file(
        get_file_path_with_randomly_chosen_file(
            const.TPS_folder_name, seed),
        2)
    print("Label: {}".format(determine_label(path, const)))
    return path


def get_file_path_with_randomly_chosen_file(folder_name, seed):
    return "{}/{}".format(folder_name, choose_random_file(folder_name, seed))


def choose_random_file(folder_name, seed):
    random.seed(seed)
    file_names = listdir(folder_name)
    return random.choice(file_names)


def convert_interface_name_to_math_text(interface_name):
    return "$MCG_{}$".format("{"+interface_name[3:]+"}")


def get_one_toy_path(folder_name, label, seed=42):
    paths, labels = read_paths_and_labels_from_pickles(folder_name)
    chosen_index = choose_random_index_where_label_matches(seed, labels, label)
    return paths[chosen_index]


def read_paths_and_labels_from_pickles(folder_name):
    paths = np.array(
        pickle.load(open("{}/paths.p".format(folder_name), "rb")))
    labels = np.array(
        pickle.load(open("{}/labels.p".format(folder_name), "rb")))
    return paths, labels


def choose_random_index_where_label_matches(seed, labels, label):
    random.seed(seed)
    return random.choice(np.where(labels == label)[0])
