from os import listdir
import glob
import numpy as np
from sklearn.utils import shuffle
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
    snapshots, snapshot_labels, snapshot_weights, snapshot_origins = \
        shuffle(
            snapshots, snapshot_labels, snapshot_weights, snapshot_origins,
            random_state=42)
    return np.array([*snapshots[:train_end]]), \
        np.array([*snapshot_labels[:train_end]]), \
        np.array([*snapshot_weights[:train_end]]), \
        np.array([*snapshots[train_end:val_end]]), \
        np.array([*snapshot_labels[train_end:val_end]]), \
        np.array([*snapshot_weights[train_end:val_end]]), \
        np.array([*snapshots[val_end:]]), \
        np.array([*snapshot_labels[val_end:]]), \
        np.array([*snapshot_weights[val_end:]])


def make_snapshots_from_paths(
        paths, path_labels, path_weights, path_origins, const):
    snapshots = []
    snapshot_labels = []
    snapshot_weights = []
    snapshot_origins = []
    for path, path_label, path_weight, path_origin in zip(
            paths, path_labels, path_weights, path_origins):
        snapshots.extend(path)
        path_len = len(path)
        snapshot_weights.extend(
            [get_snapshot_weight(path_weight, path_len)] * path_len)
        snapshot_origins.extend(
            [path_origin] * path_len)
        snapshot_labels.extend(
            [get_snapshot_label(path_label, const)] * path_len)
    snapshot_weights = normalize_snapshots_weights(snapshot_weights)
    print("Total mean weights: {}".format(np.mean(snapshot_weights)))
    print("Total sum weights: {}".format(np.sum(snapshot_weights)))
    return np.array(snapshots), \
        np.array(snapshot_labels), \
        np.array(snapshot_weights), \
        np.array(snapshot_origins)


def get_snapshot_weight(path_weight, path_len):
    return path_weight / path_len


def get_snapshot_label(path_label, const):
    if path_label == "AA":
        return const.AA_label
    elif path_label == "BB":
        return const.BB_label
    elif path_label == "AB":
        return const.AB_label
    elif path_label == "BA":
        return const.BA_label


def normalize_snapshots_weights(snapshot_weights):
    return np.array(snapshot_weights) / np.mean(snapshot_weights)


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
        return file.readlines()


def get_names_from_lines(lines, frac_len, type_function):
    """Take list of lines read from a file, keep the first fract_len
    elements, remove the end of line character at the end of each
    element and convert it to the type definded by function.
    """
    return [type_function(line[:-1]) for line in lines[:frac_len]]


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
    """Read in data file and convert into a path of the right format"""
    return round_points_to_precision(
        get_data_elements_from_lines(
            read_data_lines_from_data_file(file_path)),
        precision)


def path_outside_state_definition(path, const):
    return (snapshots_outside_state_definition(path[0], const)
            or snapshots_outside_state_definition(path[-1], const))


def snapshots_outside_state_definition(snapshot, const):
    return determine_state(snapshot, const) == "N"


def handle_path_outside_state_definition(file_name, path):
    print(("Path in {} begins (mcg = {}) or ends"
           + "(mcg = {}) outside of state definition.")
          .format(file_name, path[0][0], path[-1][0]))


def determine_label(path, const):
    start_state = determine_state(path[0], const)
    end_state = determine_state(path[-1], const)
    return start_state + end_state


def determine_state(snapshot, const):
    if snapshot[0] <= const.mcg_A:
        return "A"
    elif snapshot[0] >= const.mcg_B:
        return "B"
    else:
        return "N"


def correct_highest_interface(
        highest_interface, TIS_origins, TIS_weights, TPS_weights):
    """Update the highest TIS interface as well as the TPS weights."""
    update_factor = get_TIS_highest_interface_update_factor(
        TIS_origins, highest_interface, TPS_weights)
    TIS_full_broadcast_mask = get_TIS_highest_interface_full_broadcast_mask(
        TIS_origins, highest_interface, update_factor)
    TIS_weights = TIS_weights * TIS_full_broadcast_mask
    TPS_weights = TPS_weights * update_factor
    return TIS_weights, TPS_weights


def get_TIS_highest_interface_full_broadcast_mask(
        TIS_origins, highest_interface, update_factor):
    """Make a full mask for updating the weights of the highest interface
    without touching the other weights.
    """
    TIS_update_mask = \
        get_TIS_highest_interface_update_mask(
            TIS_origins, highest_interface, update_factor)
    TIS_non_update_mask = \
        get_TIS_highest_interface_non_update_mask(
            TIS_origins, highest_interface)
    return TIS_update_mask + TIS_non_update_mask


def get_TIS_highest_interface_update_mask(
        TIS_origins, highest_interface, update_factor):
    """Make a mask with the update_factor at all positions of the
    highest interface and 0 everywhere else.
    """
    return get_TIS_highest_interface_true_mask(
        TIS_origins, highest_interface) * update_factor


def get_TIS_highest_interface_true_mask(TIS_origins, highest_interface):
    return TIS_origins == highest_interface


def get_TIS_highest_interface_non_update_mask(
        TIS_origins, highest_interface):
    """Make a mask with 0 at all positions of the the highest interface and 1
    everywhere else.
    """
    return get_TIS_highest_interface_false_mask(
        TIS_origins, highest_interface) * 1


def get_TIS_highest_interface_false_mask(TIS_origins, highest_interface):
    return TIS_origins != highest_interface


def get_TIS_highest_interface_update_factor(
        TIS_origins, highest_interface, TPS_weights):
    """Determine by which factor the weights at the highest TIS interface
    need to be corrected.
    """
    TIS_highest_interface_cnt = \
        get_TIS_highest_interface_cnt(TIS_origins, highest_interface)
    TPS_highest_interface_cnt = \
        get_TPS_highest_interface_count(TPS_weights)
    return TIS_highest_interface_cnt \
        / (TIS_highest_interface_cnt + TPS_highest_interface_cnt)


def get_TIS_highest_interface_cnt(TIS_origins, highest_interface):
    return len([1 for origin in TIS_origins if origin == highest_interface])


def get_TPS_highest_interface_count(TPS_weights):
    return len(TPS_weights)


def read_shooting_points(filename):
    print("Read shooting point file")
    labels, shooting_points = \
        get_labels_and_shooting_points_from_shooting_data(
            get_data_elements_from_lines(
                read_data_lines_from_data_file(filename)))
    precision = 2
    shooting_points = round_points_to_precision(shooting_points, precision)
    print("{} shooting points read".format(len(labels)))
    return shooting_points, labels


def read_data_lines_from_data_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        if lines[0].startswith("#"):
            lines = lines[1:]
    return lines


def get_data_elements_from_lines(lines):
    return np.array([get_data_elements_from_line(line)
                     for line in lines])


def get_data_elements_from_line(line):
    return list(map(float, line[:-1].split(" ")[1:]))


def get_labels_and_shooting_points_from_shooting_data(shooting_data):
    labels, points = map(
        np.array, zip(*[(datum[0], datum[1:]) for datum in shooting_data]))
    return labels, points


def round_points_to_precision(points, precision):
    return np.array([round_point_to_precision(point, precision)
                    for point in points])


def round_point_to_precision(point, precision):
    return list(map(lambda x: round(x, precision), point))


def get_toy_paths(const):
    labels = const.keep_labels
    paths = [get_one_toy_path(const.toy_folder_name, label)
             for label in labels]
    return paths, labels


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


def get_TPS_and_TIS_paths(const):
    TIS_interface_names = get_TIS_interface_names(const)
    paths = [get_one_TIS_path(const=const, interface=TIS_interface_name)
             for TIS_interface_name in TIS_interface_names]\
        + [get_one_TPS_path(const=const)]
    labels = [convert_interface_name_to_math_text(TIS_interface_name)
              for TIS_interface_name in TIS_interface_names] + ["$TPS$"]
    return paths, labels


def get_TIS_interface_names(const):
    return sorted(sorted(listdir(const.TIS_folder_name)), key=len)


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
