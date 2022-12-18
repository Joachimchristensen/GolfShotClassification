import os
import struct
import numpy as np


def write_binary_data_file(output_path, data_name, samples: np.ndarray):
    with open(output_path, 'wb') as f:
        start_sample = 0
        end_sample = samples.shape[1]

        f.write(data_name.encode())
        f.write(b'\x1d')
        f.write(struct.pack(">I", 2*samples.shape[0]))
        f.write(struct.pack(">I", (end_sample - start_sample)))
        f.write(b'\x1d')

        samples_flat = np.hstack(tuple(np.hstack((samples[n, start_sample:end_sample].real, samples[n, start_sample:end_sample].imag)) for n in range(samples.shape[0])))

        f.write(struct.pack(">" + "f"*len(samples_flat), *samples_flat))


def read_header_from_binary(file_obj):
    file_obj.seek(0)
    char = file_obj.read(1)
    name = ""
    while char != b'\x1d':
        name += char.decode()
        char = file_obj.read(1)

    channels, = struct.unpack(">I", file_obj.read(4))
    n_samples, = struct.unpack(">I", file_obj.read(4))

    assert file_obj.read(1) == b'\x1d'

    return name, channels, n_samples, file_obj.tell()


def read_binary(file_path, start_sample=0, end_sample=np.inf, read_channels=None):

    with open(file_path, 'rb') as f:
        name, channels, n_samples, header_end = read_header_from_binary(f)

        if read_channels is None:
            read_channels = [c for c in range(channels)]

        if end_sample == np.inf:
            n_read = n_samples - start_sample
        else:
            n_read = end_sample - start_sample

        samples = np.empty((len(read_channels), n_read), dtype=float)

        fmt = ">" + "f"*(n_read)
        for n in read_channels:
            offset = header_end + n * n_samples * 4 + start_sample * 4
            f.seek(offset)
            samples[n, :] = struct.unpack(fmt, f.read(4*n_read))

    return name, channels, n_samples, samples


def write_preprocessed_binary_header(file_path, file_name):
    with open(file_path, 'wb') as f:
        f.write(file_name.encode())
        f.write(b'\x1d')


def write_preprocessed_binary_data(file_path, start_sample, end_sample, toi, ballvr, samples):
    with open(file_path, 'ab') as f:
        if toi is None:
            f.write(struct.pack('>d', -1.))
            f.write(struct.pack('>d', -1.))
        else:
            f.write(struct.pack('>d', toi))
            f.write(struct.pack('>d', ballvr))

        f.write(struct.pack(">I", samples.shape[0]))
        f.write(struct.pack(">I", samples.shape[1]))

        samples_flat = np.hstack((samples[n, :] for n in range(samples.shape[0])))

        f.write(struct.pack(">" + "f"*len(samples_flat), *samples_flat))


def read_preproccessed_binary_data_segment(file_path, f_index):
    with open(file_path, 'rb') as f:
        f.seek(f_index)
        toi, ball_vr, n_channels, n_samples = struct.unpack('>ddII', f.read(24))

        samples = np.empty((n_channels, n_samples), dtype=float)

        fmt = ">" + "f"*(n_samples)
        for n in range(n_channels):
            samples[n, :] = struct.unpack(fmt, f.read(4*n_samples))

        return samples, toi, ball_vr


def read_preprocessed_binary_data_indices(file_path):
    with open(file_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0, os.SEEK_SET)

        char = f.read(1)
        source_name = b""
        while char != b'\x1d':
            source_name += char
            char = f.read(1)

        indices = []
        while f.tell() < size:
            indices.append(f.tell())
            _, _, n_channels, n_samples = struct.unpack('>ddII', f.read(24))
            f.seek(4 * n_channels * n_samples, os.SEEK_CUR)

        return indices, source_name.decode()


if __name__ == "__main__":

    from trackmanlib.data.tmf_file import TMFFile
    from trackmanlib.tm_math.calculate_spectrogram import calculate_spectrogram
    from trackmanlib.plot.plot_spectrogram import plot_spectrogram

    import matplotlib.pyplot as plt

    path = r"\\FS01\Data\Development\TMAN IV\HighFPS\Outdoor\20190605 NB Vasatorp\NB Vasatorp 1.9-beta5 outdoor\Support\00106.tmf"
    name, channels, n_samples, samples = read_binary(r"C:\Projects\VisionSandbox\putting_trigger_nn\TriggerNets\TriggerNets\data\upload\020355.bin", read_channels=[0, 1, 2, 3, 4, 5, 6, 7])

    print("Get tmd")
    tmd = TMFFile(path).get_tmd_file()
    print("Get samples")
    samples = tmd.get_calibrated_samples_from_tmd()

    write_binary_data_file("test.bin", path, samples)
    print("Done writing")

    name, channels, n_samples, samples = read_binary("test.bin", read_channels=[0, 1, 2, 3, 4, 5, 6, 7])
    print("Done reading")

    for n in range(int(samples.shape[0]/2)):
        c_samples = samples[2*n, :] + 1j * samples[2*n + 1, :]

        S, P, t, f = calculate_spectrogram(c_samples, 2048, window='hanning', sample_rate=39062.5)

        plot_spectrogram(t, f, P)

    plt.show()
