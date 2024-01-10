import matplotlib.pyplot as plt
import numpy.fft
from teval import Image
from pathlib import Path
import numpy as np
from numpy import pi
from scipy.constants import epsilon_0
from teval.consts import c_thz, THz, plot_range1
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map, do_ifft, to_db
from tmm import coh_tmm as coh_tmm_full
from tmm_slim import coh_tmm
from scipy.optimize import shgo
# from teval import conductivity
from conductivity_eval import conductivity, plt_show


def thickness_analysis(path_, sample_idx_, point_=(37, -6.5)):
    img = Image(path_, sample_idx=sample_idx_)

    for d_f in [0.200, 0.400, 0.600]:
        res = conductivity(img, point_, d_film_=d_f)
        n = res["n"]

        plt.figure("n_imag")
        plt.plot(n[:, 0].real, n[:, 1].imag, label=d_f)

        n_real_fft_ = numpy.fft.rfft(n[:, 1].imag)
        freq_ = numpy.fft.rfftfreq(len(n[:, 1].imag))
        plt.figure("fft")
        plt.plot(freq_[1:], np.abs(n_real_fft_[1:]), label=d_f)


def main():
    sample_idx = 4

    # path_ = Path("/home/ftpuser/ftp/Data/HHI_Aachen/sample3/img1")

    path_ = Path(f"E:\measurementdata\HHI_Aachen\sample{sample_idx}\img1")
    path_ = Path(f"/home/ftpuser/ftp/Data/HHI_Aachen/sample{sample_idx}/img1")

    if sample_idx == 3:
        options = {"cbar_min": 1, "cbar_max": 3.0}
        # options = {"cbar_min": 0, "cbar_max": 0.030}
        # options = {"cbar_min": 0.05, "cbar_max": 0.21, "color_map": "viridis"}
        # options = {"cbar_min": 0.4, "cbar_max": 0.6, "color_map": "viridis"}
        options = {"cbar_min": 10, "cbar_max": 60, "color_map": "viridis"}
        # options = {"cbar_min": 0, "cbar_max": 15, "color_map": "viridis"}
    else:
        options = {"cbar_min": 1, "cbar_max": 3.0, "log_scale": True}
        options = {"cbar_min": 0, "cbar_max": 0.015}
        options = {"cbar_min": 0, "cbar_max": 1.5}
        # options = {"cbar_min": 0.05, "cbar_max": 0.21, "color_map": "viridis"}
        options = {"cbar_min": 10, "cbar_max": 60, "color_map": "viridis"}
        # options = {"cbar_min": -25, "cbar_max": 25, "color_map": "viridis"}
        # options = {"cbar_min": 9, "cbar_max": 12, "color_map": "viridis"}

    img = Image(path_, options=options, sample_idx=sample_idx)
    # img.plot_image()
    # point = (31.0, 4)
    point = (30.0, -8.5)  # doesnt work
    point = (38.0, 0.5)  # doesnt work
    #point = (28.0, -4.0)  # doesnt work
    #point = (40.50, -1.5)  # doesnt work
    #point = (29.50, -3.0)  # doesnt work
    #point = (31.0, 4.0)  # doesnt work
    point = (35.5, -1.5)  # doesnt work
    point = (35.0, 7.0)
    point = (38.5, -3.5)
    # point = (30.5, -8.5)  # works
    # img.system_stability(selected_freq_=2.000)
    # img.plot_point(*point)
    img.plot_image(quantity="conductivity", selected_freq=1.750)
    # img.plot_image(quantity="meas_time_delta")
    # img.plot_image(quantity="power", selected_freq=(1.95, 2.05))
    # img.plot_image(quantity="amplitude_transmission", selected_freq=1.000)

    # plt_show()

    # thickness_analysis(path_, sample_idx)
    res = conductivity(img, point)
    if isinstance(res, dict):
        n = res["n"]
    else:
        n = res

    plt.figure("n")
    plt.plot(n[:, 0].real, n[:, 1].real, label="n real")
    plt.plot(n[:, 0].real, n[:, 1].imag, label="n imag")
    # plt.ylim((0, 0.02))

    sheet_cond = res["tinkham"]
    plt.figure("tinkham")
    plt.plot(sheet_cond[:, 0].real, sheet_cond[:, 1].real, label="sheet conductivity tinkham real")
    plt.plot(sheet_cond[:, 0].real, sheet_cond[:, 1].imag, label="sheet conductivity imag")

    plt_show()


if __name__ == '__main__':
    main()
