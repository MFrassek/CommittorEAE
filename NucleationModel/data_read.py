from os import listdir
import glob
import numpy as np
from sklearn.utils import shuffle


def make_train_val_test_from_RPE_and_TPS(const):
    return make_train_val_test_split_snapshots_from_snapshots(
        *make_snapshots_from_paths(
            *filter_paths(
                *make_paths_from_RPE_and_TPS_data(
                    const),
                const),
            const),
        const)


def make_train_val_test_split_snapshots_from_snapshots(
        snapshots,
        snapshot_labels,
        snapshot_weights,
        const):
    train_end = int(len(snapshots) * const.train_ratio)
    val_end = train_end + int(len(snapshots) * const.val_ratio)
    snapshots, snapshot_labels, snapshot_weights = \
        shuffle(
            snapshots,
            snapshot_labels,
            snapshot_weights,
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


def make_snapshots_from_paths(paths, path_labels, path_weights, const):
    snapshots = []
    snapshot_labels = []
    snapshot_weights = []
    for path_nr, path in enumerate(paths):
        path_label = path_labels[path_nr]
        path_weight = path_weights[path_nr]
        for snapshot_nr, snapshot in enumerate(path):
            # Iterate over all indices within each path and appends
            # accordingly the snapshot as well as label
            # and weight.
            snapshots.append(snapshot)
            snapshot_weights.append(path_weight)
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
    print("Mean weights: {}".format(np.mean(snapshot_weights)))
    return np.array(snapshots), \
        np.array(snapshot_labels), \
        np.array(snapshot_weights)


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


def filter_paths(paths, path_labels, path_weights, const):
    filtered_tuple_list = [(path, label, weight) for (path, label, weight)
                           in zip(paths, path_labels, path_weights)
                           if label in const.keep_labels]
    return list(map(list, zip(*filtered_tuple_list)))


def make_paths_from_RPE_and_TPS_data(const):
    print("Read RPE files")
    RPE_paths, RPE_labels, RPE_weights, RPE_names = \
        make_paths_from_RPE_data(const)
    print("Read TPS files")
    # Read in the TPS files and generate paths, labels, weights and names.
    # Weights are chosen based on the minimal weight assigned to the RPE paths.
    TPS_paths, TPS_labels, TPS_weights, TPS_names = \
        make_paths_from_TPS_data(min(RPE_weights), const)
    weights = np.append(RPE_weights, TPS_weights, axis=0)
    weights = weights/np.mean(weights)
    # Return the merges  RPE and TPS arrays
    return np.append(RPE_paths, TPS_paths, axis=0), \
        np.append(RPE_labels, TPS_labels, axis=0), \
        weights


def make_paths_from_RPE_data(const):
    foldername = const.RPE_folder_name
    paths = []
    labels = []
    mc_weights = []
    reweights = []
    names = []
    for folder in listdir(foldername):
        print(folder)
        # for each folder opens the file path_name or path_name.txt as names
        # glob.glob assures that both file nnp.amings will be accepted
        with open(glob.glob("{}/{}/path_name*"
                            .format(foldername, folder))[0], "r") as name_file:
            # similarly opens the corresponding path_weights file as weights
            with open(glob.glob("{}/{}/path_weight*"
                                .format(foldername, folder))[0], "r") \
                                    as weight_file:
                with open(glob.glob("{}/{}/rpe_weigh*"
                                    .format(foldername, folder))[0], "r") \
                                    as reweight_file:

                    name_lines = name_file.readlines()
                    weight_lines = weight_file.readlines()
                    reweight_lines = reweight_file.readlines()
                    name_lines, weight_lines, reweight_lines = \
                        shuffle(
                            name_lines, weight_lines,
                            reweight_lines, random_state=42)
                    frac_len = int(len(name_lines) * const.used_RPE_frac)
                    print("Total paths: {}\t Used paths: {}"
                          .format(len(name_lines), frac_len))
                    path_names = list(map(
                        lambda x: x[:-1],
                        name_lines[:frac_len]))
                    weight_names = list(map(
                        lambda x: int(x[:-1]),
                        weight_lines[:frac_len]))
                    reweight_names = list(map(
                        lambda x: float(x[:-1]),
                        reweight_lines[:frac_len]))
                    for file_nr in range(frac_len):
                        with open(
                                "{}/{}/light_data/{}".format(
                                    foldername,
                                    folder,
                                    path_names[file_nr]),
                                "r") as path:
                            path = path.readlines()
                            # iterates over all snapshots in the trajectory
                            # removes the linebreak character at the end ("\n")
                            # splits them along all occurences of " "
                            # drops the first column (snapshot_index)
                            # transforms the strings into floats
                            path = np.array([list(map(float, snap[:-1]
                                            .split(" ")[1:]))[:17]
                                            for snap in path])
                            if path_outside_state_definition(path, const):
                                print(("Path in {} begins (mcg = {}) or ends"
                                       + "(mcg = {}) outside of state"
                                       + " definition.")
                                      .format(
                                        path_names[file_nr],
                                        path[0][0],
                                        path[-1][0]))
                            else:
                                paths.append(path)
                                labels.append(determine_label(path, const))
                                mc_weights.append(weight_names[file_nr])
                                reweights.append(reweight_names[file_nr])
                                names.append("{}_{}"
                                             .format(
                                                folder,
                                                path_names[file_nr][:-4]))

    # Multiply the two weight lists.
    weights = np.array(mc_weights) * np.array(reweights)
    # Normalize the weights, so that their mean equals 1.
    print("Sum weights: {}".format(sum(weights)))
    print("Mean weights: {}".format(np.mean(weights)))
    return np.array(paths), np.array(labels), \
        np.array(weights), np.array(names)


def make_paths_from_TPS_data(TPS_weight, const):
    foldername = const.TPS_folder_name
    paths = []
    labels = []
    names = []
    precision = 2
    for file in listdir(foldername):
        with open("{}/{}".format(foldername, file), "r") as file_name:
            file_name.readline()
            path = file_name.readlines()
            # Iterate over all snapshots in the trajectory
            # remove the linebreak character at the end ("\n")
            # split them along all occurences of " "
            # drop the first column (snapshot_index)
            # transform the strings into floats
            # and round to the given precision.
            path = np.array(
                [list(map(lambda x: round(float(x), precision),
                 snap[:-1].split(" ")[1:]))[:17] for snap in path])
            if path_outside_state_definition(path, const):
                print(("Path in {} begins (mcg = {}) or ends"
                       + "(mcg = {}) outside of state definition.")
                      .format(
                        file,
                        path[0][0],
                        path[-1][0]))
            else:
                paths.append(path)
                labels.append(determine_label(path, const))
                names.append(file[:3])

    frac_len = int(len(paths) * const.used_TPS_frac)
    print("Total paths: {}\t Used paths: {}".format(len(paths), frac_len))
    weights = [TPS_weight for i in range(frac_len)]
    paths, labels, names = shuffle(paths, labels, names, random_state=42)
    print(sum(weights))
    return np.array(paths)[:frac_len], np.array(labels)[:frac_len], \
        np.array(weights), np.array(names)[:frac_len]


def path_outside_state_definition(path, const):
    return (snapshots_outside_state_definition(path[0], const)
            or snapshots_outside_state_definition(path[-1], const))


def snapshots_outside_state_definition(snapshot, const):
    return snapshot[0] > const.mcg_A and snapshot[0] < const.mcg_B


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
