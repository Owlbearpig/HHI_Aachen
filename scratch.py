from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, speed_of_light
from scipy import signal
from scipy.optimize import shgo
import logging


def window(data_td, win_len=None, win_start=None, shift=None, en_plot=False, slope=0.50):
    t, y = data_td[:, 0], data_td[:, 1]
    t -= t[0]
    y -= np.mean(data_td[:10, 1])

    pulse_width = 10  # ps
    dt = np.mean(np.diff(t))

    if win_len is None:
        win_len = int(pulse_width / dt)
    else:
        win_len = int(win_len / dt)

    if win_len > len(y):
        win_len = len(y)

    if win_start is None:
        win_center = np.argmax(np.abs(y))
        win_start = win_center - int(win_len / 2)
    else:
        win_start = int(win_start / dt)

    if win_start < 0:
        win_start = 0

    pre_pad = np.zeros(win_start)
    window_arr = signal.windows.tukey(win_len, slope)
    post_pad = np.zeros(len(y) - win_len - win_start)

    window_arr = np.concatenate((pre_pad, window_arr, post_pad))

    if shift is not None:
        window_arr = np.roll(window_arr, int(shift / dt))

    y_win = y * window_arr

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label="Sam. before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_arr, label="Window")
        plt.plot(t, y_win, label="Sam. after windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    return np.array([t, y_win]).T


def rel_meas_time(filename):
    filename = filename.name
    if isinstance(filename, Path):
        filename = str(filename)

    date_str = "-".join(filename.split("-")[:5])

    seconds = (datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f") - datetime.min).total_seconds()

    return seconds


def calc_phase(f_axis, data_fd):
    phi = np.unwrap(np.angle(data_fd))
    slice_ = (0.25 < f_axis) * (f_axis < 1.0)
    res = np.polyfit(f_axis[slice_], phi[slice_], 1)

    return phi


"""
base_path_sub = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_24/GaAs Wafer 25")
base_path_film = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_24/GaAs_Te 19075")

ref_sub_files = list((base_path_sub / "Reference").glob("*.txt"))
sam_sub_files = list((base_path_sub / "Sample").glob("*.txt"))
ref_film_files = list((base_path_film / "Reference").glob("*.txt"))
sam_film_files = list((base_path_film / "Sample").glob("*.txt"))
"""

"""
base_path_sub = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_24/GaAs Wafer 25")
base_path_film = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_24/GaAs_Te 19073")

ref_sub_files = list((base_path_sub / "Reference").glob("*.txt"))
sam_sub_files = list((base_path_sub / "Sample").glob("*.txt"))
ref_film_files = list((base_path_film / "Reference").glob("*.txt"))
sam_film_files = list((base_path_film / "Sample").glob("*.txt"))
"""

base_path_sub = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_09/GaAs_Wafer_25")
base_path_film = Path(r"/home/ftpuser/ftp/Data/THzConductivity/MarielenaData/2021_08_09/GaAs_C doped")

ref_sub_files = list((base_path_sub / "Reference").glob("*.txt"))
sam_sub_files = list((base_path_sub / "Sample").glob("*.txt"))
ref_film_files = list((base_path_film / "Reference").glob("*.txt"))
sam_film_files = list((base_path_film / "Sample").glob("*.txt"))

"""
base_path = Path(r"/home/ftpuser/ftp/Data/SemiconductorSamples/GaAaTe_wafer_sam Reameasure")
all_files = list(base_path.glob("*.txt"))

ref_sub_files, sam_sub_files, ref_film_files, sam_film_files = [], [], [], []
for file in all_files:
    if "reference" in str(file):
        ref_sub_files.append(file)
    elif "substrate" in str(file):
        sam_sub_files.append(file)
    elif "hole" in str(file):
        ref_film_files.append(file)
    else:
        sam_film_files.append(file)
"""
"""
base_path = Path(r"/home/ftpuser/ftp/Data/SemiconductorSamples/Wafer_25_and_wafer_19073")
all_files = list(base_path.glob("*.txt"))

ref_sub_files, sam_sub_files, ref_film_files, sam_film_files = [], [], [], []
for file in all_files:
    if "reference" in str(file).lower():
        ref_sub_files.append(file)
    elif "samsub" in str(file).lower():
        sam_sub_files.append(file)
    elif "empty" in str(file).lower():
        ref_film_files.append(file)
    else:
        sam_film_files.append(file)
"""

"""
base_path = Path(r"/home/ftpuser/ftp/Data/HHI_Aachen/single_points_02_10_2024/x_37")
all_files = list(base_path.glob("*.txt"))

ref_sub_files, sam_sub_files, ref_film_files, sam_film_files = [], [], [], []
for file in all_files:
    if "ref" in str(file):
        ref_sub_files.append(file)
    elif "sub" in str(file):
        sam_sub_files.append(file)
    elif "sam" in str(file):
        sam_film_files.append(file)
    else:
        ref_film_files.append(file)
"""

if not len(ref_film_files) > 0:
    ref_film_files = ref_sub_files

ref_sub_files = sorted(ref_sub_files, key=lambda x: rel_meas_time(x))
sam_sub_files = sorted(sam_sub_files, key=lambda x: rel_meas_time(x))
ref_film_files = sorted(ref_film_files, key=lambda x: rel_meas_time(x))
sam_film_files = sorted(sam_film_files, key=lambda x: rel_meas_time(x))

l = 700 * 1e-9
l = 700 * 1e-7
n1, n3 = 1, 3.68

phi_ref, phi_sam = [], []


def to_db(data_fd):
    return 20 * np.log10(np.abs(data_fd))


def fft(data_td):
    return np.conj(np.fft.rfft(data_td))
    return np.fft.rfft(data_td)


scale = 1  # 1.27e3


def drude(freq_axis, tau, sig0):
    sig0 *= scale
    w = 2 * np.pi * freq_axis
    return sig0 / (1 - 1j * tau * 1e-3 * w)


def rand_ints(l_):
    ret = []
    for _ in range(3):
        ret.append(np.random.randint(0, len(l_)))
    return sorted(ret)


idx_list = rand_ints(ref_sub_files)

fig, (ax0, ax1) = plt.subplots(2, 1, num="1")
ax0.set_title("real")
ax1.set_title("imag")

ax0.set_ylim((-50, 300))
ax1.set_ylim((-100, 100))

chosen_idx = 4
sigmas = []
t = (ref_sub_files, sam_sub_files, ref_film_files, sam_film_files)
for i, (ref_sub_file, sam_sub_file, ref_film_file, sam_film_file) in enumerate(zip(*t)):
    # if i not in [100, 200, 1000, 1500, 2000]:
    # print(ref_sub_file, sam_sub_file, ref_film_file, sam_film_file, sep="\n")

    ref_sub_td, sam_sub_td = np.loadtxt(ref_sub_file), np.loadtxt(sam_sub_file)
    ref_film_td, sam_film_td = np.loadtxt(ref_film_file), np.loadtxt(sam_film_file)

    ref_sub_td = window(ref_sub_td)
    sam_sub_td = window(sam_sub_td)
    ref_film_td = window(ref_film_td)
    sam_film_td = window(sam_film_td)

    ref_sub_fd, sam_sub_fd = fft(ref_sub_td[:, 1]), fft(sam_sub_td[:, 1]),
    ref_film_fd, sam_film_fd = fft(ref_film_td[:, 1]), fft(sam_film_td[:, 1])

    freq = np.fft.rfftfreq(len(ref_sub_td[:, 0]), d=0.05)
    mask = (0.25 <= freq) * (freq < 3.0)

    phi_ref_sub = calc_phase(freq, ref_sub_fd)
    phi_sam_sub = calc_phase(freq, sam_sub_fd)
    phi_ref_film = calc_phase(freq, ref_film_fd)
    phi_sam_film = calc_phase(freq, sam_film_fd)

    ref_sub_fd = np.abs(ref_sub_fd) * np.exp(1j * phi_ref_sub)
    sam_sub_fd = np.abs(sam_sub_fd) * np.exp(1j * phi_sam_sub)
    ref_film_fd = np.abs(ref_film_fd) * np.exp(1j * phi_ref_film)
    sam_film_fd = np.abs(sam_film_fd) * np.exp(1j * phi_sam_film)
    """
    plt.figure("time domain")
    t = ref_sub_td[:, 0]
    plt.plot(t, ref_sub_td[:, 1], label="sub ref")
    plt.plot(t, sam_sub_td[:, 1], label="sub sam")
    plt.plot(t, ref_film_td[:, 1], label="film ref")
    plt.plot(t, sam_film_td[:, 1], label="film sam")
    plt.legend()

    plt.figure("Spectrum")
    plt.plot(freq[mask], to_db(ref_sub_fd[mask]), label="sub ref")
    plt.plot(freq[mask], to_db(sam_sub_fd[mask]), label="sub sam")
    plt.plot(freq[mask], to_db(ref_film_fd[mask]), label="film ref")
    plt.plot(freq[mask], to_db(sam_film_fd[mask]), label="film sam")
    plt.legend()

    plt.figure("Phase")
    plt.plot(freq[mask], phi_ref_sub[mask], label="sub ref")
    plt.plot(freq[mask], phi_sam_sub[mask], label="sub sam")
    plt.plot(freq[mask], phi_ref_film[mask], label="film ref")
    plt.plot(freq[mask], phi_sam_film[mask], label="film sam")
    plt.legend()
    """
    T_sub, T_film = sam_sub_fd / ref_sub_fd, sam_film_fd / ref_film_fd

    sigma = ((T_sub / T_film - 1) * epsilon_0 * speed_of_light * (n1 + n3) / l) / scale

    # if i in idx_list:
    if i == chosen_idx:
        ax0.plot(freq[mask], sigma.real[mask], label=f"measurement {i}")
        ax1.plot(freq[mask], sigma.imag[mask], label=f"measurement {i}")

    sigmas.append(sigma)

    """
    def opt_fun(x):
        real_part = (drude(freq, *x).real - sigma.real) ** 2
        imag_part = (drude(freq, *x).imag - sigma.imag) ** 2

        return np.sum(real_part[mask] + imag_part[mask]) / (1000 * len(freq[mask]))

    logging.basicConfig(level=logging.INFO)
    opt_res = shgo(opt_fun, bounds=[(0.1, 0.5), (0.1, 0.5)], n=2000, iters=10,
                   minimizer_kwargs={"method": "Nelder-Mead",
                                     "options": {"maxev": np.inf, "maxiter": 20000,
                                                 "tol": 1e-16, "fatol": 1e-16, "xatol": 1e-16, }
                                     },
                   options={"maxfev": np.inf, "f_tol": 1e-16, "maxiter": 20000, "ftol": 1e-16, "xtol": 1e-16,
                            "maxev": 20000, "minimize_every_iter": True,
                            "disp": True},
                   )
    print(opt_res)
    print(opt_fun(opt_res.x))
    print(opt_fun([0.1, 0.1]))
    sigma_drude_fit = drude(freq, *opt_res.x)
    # freq = np.logspace(0, 10, 1000)
    
    
    ax0.plot(freq[mask], sigma_drude_fit.real[mask], label="drude fit")
    ax1.plot(freq[mask], sigma_drude_fit.imag[mask], label="drude fit")
    """

    # plt.semilogx(freq, drude(freq, tau=10**-5, sig0=1).real, label="drude")
    # plt.semilogx(freq, drude(freq, tau=10**-5, sig0=1).imag, label="drude")

sigmas = np.array(sigmas)
sigma_avg = np.mean(sigmas, axis=0)

ax0.plot(freq[mask], sigma_avg.real[mask], label=f"average")
ax1.plot(freq[mask], sigma_avg.imag[mask], label=f"average")

def opt_fun(x):
    real_part = (drude(freq, *x).real - sigmas[chosen_idx].real) ** 2
    imag_part = (drude(freq, *x).imag - sigmas[chosen_idx].imag) ** 2

    return np.sum(real_part[mask] + imag_part[mask]) / (len(freq[mask]) * 1000)


logging.basicConfig(level=logging.INFO)
opt_res = shgo(opt_fun, bounds=[(10, 300), (10, 2000)], n=2000, iters=100,
               minimizer_kwargs={"method": "Nelder-Mead",
                                 "options": {"maxev": np.inf, "maxiter": 40000,
                                             "tol": 1e-16, "fatol": 1e-16, "xatol": 1e-16, }
                                 },
               options={"maxfev": np.inf, "f_tol": 1e-16, "maxiter": 40000, "ftol": 1e-16, "xtol": 1e-16,
                        "maxev": 40000, "minimize_every_iter": True,
                        "disp": True},
               )
print(opt_res)
print(opt_fun(opt_res.x))
print(opt_fun([111, 583]))
sigma_drude_fit = drude(freq, *opt_res.x)
# freq = np.logspace(0, 10, 1000)

ax0.plot(freq[mask], sigma_drude_fit.real[mask], label="drude fit new")
ax1.plot(freq[mask], sigma_drude_fit.imag[mask], label="drude fit new")

sigma_drude = drude(freq, tau=95.2, sig0=520.7)
#ax0.plot(freq[mask], sigma_drude.real[mask], label="drude")
#ax1.plot(freq[mask], sigma_drude.imag[mask], label="drude")

sigma_drude = drude(freq, tau=111, sig0=583.2)
#ax0.plot(freq[mask], sigma_drude.real[mask], label="drude fit")
#ax1.plot(freq[mask], sigma_drude.imag[mask], label="drude fit")

ax0.legend()
ax1.legend()

plt.legend()
plt.show()
