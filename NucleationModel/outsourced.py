shooting_points, shooting_labels = read_shooting_points(
    "total_data_till_982mc_280K.txt")

shootingData = DatasetData(
    shooting_points,
    shooting_points,
    shooting_labels,
    np.ones(len(shooting_labels)),
    flag="Shooting")
shooting_grid_past_snapshots, \
    shooting_norm_past_snapshots, \
    shooting_pB_dict, \
    shooting_pBs, \
    shooting_pB_weights = pipeline.rbnga(shootingData)
shooting_trimmed_grid_past_snapshots, \
    shooting_trimmed_norm_past_snapshots, \
    shooting_trimmed_labels, \
    shooting_trimmed_weights, \
    shooting_trimmed_pB_dict, \
    shooting_trimmed_pBs, \
    shooting_trimmed_pB_weights, \
    shooting_trimmed_balanced_pB_weights = pipeline.rbngatb(shootingData)


def make_xy_linspaces(pipeline, x_int, y_int):
    xs = np.linspace(
        pipeline.lower_bound[x_int],
        pipeline.upper_bound[x_int],
        11)
    ys = np.array([68.14 - 0.4286*x for x in xs])
    xs = (xs - pipeline.mean[0]) / pipeline.std[0]
    ys = (ys - pipeline.mean[1]) / pipeline.std[1]
    xs = (xs - pipeline.minima[0]) / (pipeline.maxima[0] - pipeline.minima[0])\
        * c.resolution
    ys = (ys - pipeline.minima[1]) / (pipeline.maxima[1] - pipeline.minima[1])\
        * c.resolution
    ys = 2*c.resolution - ys
    return np.array(xs), np.array(ys)


def plot_one_map(label_map, stamp):
    plt.figure()
    plt.plot(
        *make_xy_linspaces(pipeline, 0, 1),
        c="r")
    plt.imshow(
        np.transpose(label_map[0])[::-1],
        cmap=make_halfpoint_divided_colormap(c.logvmin),
        interpolation='nearest',
        norm=mpl.colors.LogNorm(
            vmin=c.logvmin,
            vmax=1.0))
    plt.colorbar()
    plt.tick_params(
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False)
    plt.savefig("results/{}.png".format(stamp))
    plt.show()


def compare_dividing_surfaces():
    train_label_map = Plotter.calc_map_given(
        x_pos=0,
        y_pos=1,
        resolution=c.resolution,
        grid_snapshots=train_grid_past_snapshots,
        labels=trainData.labels,
        weights=trainData.weights)

    shooting_label_map = Plotter.calc_map_given(
        x_pos=0,
        y_pos=1,
        resolution=c.resolution,
        grid_snapshots=shooting_grid_past_snapshots,
        labels=shootingData.labels,
        weights=shootingData.weights)

    plot_one_map(
        train_label_map,
        "DividingSurfaceTest_trainData_" + c.data_stamp)
    plot_one_map(
        shooting_label_map,
        "DividingSurfaceTest_shootingData_" + c.data_stamp)


compare_dividing_surfaces()

######


def get_shooting_maps(pre_stamp, labels, weights):
    super_map = Plotter.plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        const=c,
        pre_stamp="shooting_"+pre_stamp,
        method=Plotter.calc_map_given,
        grid_snapshots=shooting_grid_past_snapshots,
        labels=labels,
        weights=weights)
    return super_map


def compare_super_maps(map1, map2):
    are_equal = True
    unequal_counter = 0
    above_counter = 0
    for y in range(len(map1)):
        for x in range(len(map1[y])):
            for z in range(len(map1[y][x])):
                for i in range(len(map1[y][x][z][z])):
                    for j in range(len(map1[y][x][z][z][i])):
                        if map1[y][x][z][z][i][j] != map2[y][x][z][z][i][j] \
                            and not (np.isnan(map1[y][x][z][z][i][j])
                                     and np.isnan(map2[y][x][z][z][i][j])):
                            are_equal = False
                            unequal_counter += 1
                            if map1[y][x][z][z][i][j] - map2[y][x][z][z][i][j]\
                               > 0.01:
                                above_counter += 1
    return are_equal, unequal_counter, above_counter

#####


def show_batch(dataset):
    for batch, label, weights in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
