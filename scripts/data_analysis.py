import numpy as np
import matplotlib.pyplot as plt


def plot_class_mean_windows():

    nt_means, nt_stds = np.load('nt_means.npy'), np.load('nt_stds.npy')
    putt_means, putt_stds = np.load('putt_means.npy'), np.load('putt_stds.npy')
    fs_means, fs_stds = np.load('fs_means.npy'), np.load('fs_stds.npy')
    N = 4096
    #plt.interactive(False)
    fig, axs = plt.subplots(4, 2, sharex=True, sharey=False)
    fig.suptitle('Window averages\n')
    i = 0
    for row, column in zip([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]):
        nt_mean, putt_mean, fs_mean = nt_means[i, :], putt_means[i, :], fs_means[i, :]
        nt_std, putt_std, fs_std = nt_stds[i, :], putt_stds[i, :], fs_stds[i, :]

        axs[row, column].plot(range(N), nt_mean, '-.', c='b', label='No trigger', linewidth=0.1)
        # axs[row, column].fill_between(range(N), nt_mean-nt_std, nt_mean+nt_std, alpha=0.2)

        axs[row, column].plot(range(N), putt_mean, '-', c='r', label='Putt', linewidth=0.1)
        # axs[row, column].fill_between(range(N), putt_mean-putt_std, putt_mean+putt_std, alpha=0.2)

        axs[row, column].plot(range(N), fs_mean, '--', c='g', label='FullSwing', linewidth=0.1)
        # axs[row, column].fill_between(range(N), fs_mean-fs_std, fs_mean+fs_std, alpha=0.2)

        axs[row, column].set_title(f'Channel {i+1}')
        if i == 0:
            axs[row, column].legend(prop={'size': 6}, loc='right')

        i += 1
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    plot_class_mean_windows()

    nt_means, nt_stds = np.load('nt_means.npy'), np.load('nt_stds.npy')
    putt_means, putt_stds = np.load('putt_means.npy'), np.load('putt_stds.npy')
    fs_means, fs_stds = np.load('fs_means.npy'), np.load('fs_stds.npy')
    N = 4096
