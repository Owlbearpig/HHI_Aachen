import matplotlib.pyplot as plt


def plt_show():
    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        # save_fig(fig_label)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()

