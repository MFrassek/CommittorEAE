from os import listdir
import glob
import numpy as np
from sklearn.utils import shuffle


def read_RPE(
        foldername, A_mcg_below, B_mcg_above,
        C_cage_big_below, used_frac):
    """"""
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
                    frac_len = int(len(name_lines)*used_frac)
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
                            if ((path[0][0] > A_mcg_below and
                                path[0][0] < B_mcg_above) or
                                (path[-1][0] > A_mcg_below and
                                 path[-1][0] < B_mcg_above)):
                                print(("Path in {} begins (mcg = {}) or ends"
                                       + "(mcg = {}) outside of state"
                                       + " definition.")
                                      .format(
                                       path_names[file_nr],
                                       path[0][0],
                                       path[-1][0]))
                            else:
                                if path[0][0] <= A_mcg_below:
                                    if path[-1][0] <= A_mcg_below:
                                        label = "AA"
                                    elif path[-1][0] >= B_mcg_above:
                                        if np.amax(path, axis=0)[8] \
                                                >= C_cage_big_below:
                                            label = "AB"
                                        else:
                                            label = "AC"
                                elif path[0][0] >= B_mcg_above:
                                    if path[-1][0] <= A_mcg_below:
                                        label = "BA"
                                    elif path[-1][0] >= B_mcg_above:
                                        label = "BB"
                                paths.append(path)
                                labels.append(label)
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


def read_TPS(
        foldername, A_mcg_below, B_mcg_above,
        C_cage_big_below, used_frac, TPS_weight):
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
            if ((path[0][0] > A_mcg_below and
                 path[0][0] < B_mcg_above) or
                (path[-1][0] > A_mcg_below and
                 path[-1][0] < B_mcg_above)):
                print(("Path in {} begins (mcg = {}) or ends"
                       + "(mcg = {}) outside of state definition.")
                      .format(
                       file,
                       path[0][0],
                       path[-1][0]))
                label = "N"
            else:
                if path[0][0] <= A_mcg_below:
                    if path[-1][0] <= A_mcg_below:
                        label = "AA"
                    elif path[-1][0] >= B_mcg_above:
                        if np.amax(path, axis=0)[8] \
                                >= C_cage_big_below:
                            label = "AB"
                        else:
                            label = "AC"
                elif path[0][0] >= B_mcg_above:
                    if path[-1][0] <= A_mcg_below:
                        label = "BA"
                    elif path[-1][0] >= B_mcg_above:
                        label = "BB"
                paths.append(path)
                labels.append(label)
                names.append(file[:3])

    frac_len = int(len(paths) * used_frac)
    print("Total paths: {}\t Used paths: {}".format(len(paths), frac_len))
    weights = [TPS_weight for i in range(frac_len)]
    paths, labels, names = shuffle(paths, labels, names, random_state=42)
    print(sum(weights))
    return np.array(paths)[:frac_len], np.array(labels)[:frac_len], \
        np.array(weights), np.array(names)[:frac_len]


def read_RPE_and_TPS(
        RPE_foldername, TPS_foldername,
        A_mcg_below, B_mcg_above,
        C_cage_big_below, used_RPE_frac, used_TPS_frac):
    print("Read RPE files")
    RPE_paths, RPE_labels, RPE_weights, RPE_names = \
        read_RPE(
            RPE_foldername, A_mcg_below, B_mcg_above,
            C_cage_big_below, used_RPE_frac)
    print("Read TPS files")
    # Read in the TPS files and generate paths, labels, weights and names.
    # Weights are chosen based on the minimal weight assigned to the RPE paths.
    TPS_paths, TPS_labels, TPS_weights, TPS_names = \
        read_TPS(
            TPS_foldername, A_mcg_below, B_mcg_above,
            C_cage_big_below, used_TPS_frac, min(RPE_weights))
    weights = np.append(RPE_weights, TPS_weights, axis=0)
    weights = weights/np.mean(weights)
    # Return the merges  RPE and TPS arrays
    return np.append(RPE_paths, TPS_paths, axis=0), \
        np.append(RPE_labels, TPS_labels, axis=0), \
        weights,\
        np.append(RPE_names, TPS_names, axis=0)


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
