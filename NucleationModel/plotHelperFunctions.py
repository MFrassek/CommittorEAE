from plotter import Plotter


def plot_loss_history(history, file_name):
    plt.figure(figsize=(8, 8))
    for key, log_loss in history.history.items():
        plt.plot(range(1, 1+len(log_loss)), log_loss, label=key)
        plt.scatter(range(1, 1+len(log_loss)), log_loss, s=10)
    plt.ylim(0,)
    plt.xlim(1, len(log_loss))
    plt.xticks(range(1, 1+len(log_loss)))
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.show()


def plot_ground_truth(
        reduced_list_var_names, reduced_name_to_list_position,
        pipeline, const, grid_snapshots, labels, weights, pre_stamp):
    Plotter.plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        method=Plotter.calc_map_given,
        grid_snapshots=grid_snapshots,
        labels=labels,
        weights=weights)
