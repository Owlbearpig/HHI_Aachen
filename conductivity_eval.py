import numpy as np
from numpy import pi
from scipy.constants import epsilon_0
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map, do_ifft, to_db
from teval.consts import c_thz, THz, plot_range1, c0
from tmm import coh_tmm as coh_tmm_full
from tmm_slim import coh_tmm
import matplotlib.pyplot as plt
from scipy.optimize import shgo


def plt_show():
    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        # save_fig(fig_label)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()


d_sub = 1000
freq_range = (0.25, 2.5)
angle_in = 0.0
initial_shgo_iters = 3


def interpolate_pulse(data_td):
    pass


def sub_refidx_a(img_, point):
    # img_.plot_point(*point)
    # plt.show()
    sub_meas = img_.get_measurement(*point)
    sam_td = sub_meas.get_data_td()
    ref_td = img_.get_ref(point=point)

    plt_show()

    sam_td = window(sam_td, en_plot=False, slope=0.99)
    ref_td = window(ref_td, en_plot=False, slope=0.99)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    freqs = ref_fd[:, 0].real
    omega = 2 * np.pi * freqs

    phi_ref = phase_correction(ref_fd, fit_range=(0.1, 1.2))
    phi_sam = phase_correction(sam_fd, fit_range=(0.1, 1.2))

    phi_diff = phi_sam[:, 1] - phi_ref[:, 1]

    n0 = 1 + c_thz * phi_diff / (omega * d_sub)
    k0 = -c_thz * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1]) * (1 + n0) ** 2 / (4 * n0)) / (omega * d_sub)

    return np.array([freqs, n0 + 1j * k0], dtype=complex).T


def sub_refidx_tmm(img_, point):
    en_plot = True
    # img_.plot_point(*point)
    # plt.show()

    sub_meas = img_.get_measurement(*point)
    sam_td = sub_meas.get_data_td()
    ref_td = img_.get_ref(point=point)

    sam_td[:, 0] -= sam_td[0, 0]

    # sam_td = window(sam_td, en_plot=False, win_len=10)
    # ref_td = window(ref_td, en_plot=False, win_len=10)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)
    phi = np.angle(sam_fd[:, 1] / ref_fd[:, 1])

    freqs = ref_fd[:, 0].real

    zero = np.zeros_like(freqs, dtype=complex)
    one = np.ones_like(freqs, dtype=complex)
    omega = 2 * np.pi * freqs

    f_opt_idx = f_axis_idx_map(freqs, freq_range)

    d_list = np.array([np.inf, d_sub, np.inf], dtype=float)

    phase_shift = np.exp(-1j * d_sub * omega / c_thz)

    # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

    def calc_model(n_model, ret_t=False, ret_T_and_R=False):
        n_list_ = np.array([one, n_model, one], dtype=complex).T

        R = np.zeros_like(freqs, dtype=complex)
        T = np.zeros_like(freqs, dtype=complex)
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx_, freq_ in enumerate(freqs):
            if np.isclose(freq_, 0):
                continue
            lam_vac = c_thz / freq_
            n = n_list_[f_idx_]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac) * phase_shift[f_idx_]
            ts_tmm_fd[f_idx_] = t_tmm_fd
            if ret_T_and_R:
                dict_res = coh_tmm_full("s", n, d_list, angle_in, lam_vac)
                T[f_idx_] = dict_res["T"]
                R[f_idx_] = dict_res["R"]

        sam_tmm_fd_ = np.array([freqs, ts_tmm_fd * ref_fd[:, 1]]).T
        sam_tmm_td_ = do_ifft(sam_tmm_fd_)
        sam_tmm_td_[:, 0] -= sam_tmm_td_[0, 0]

        if ret_T_and_R:
            return T, R
        if ret_t:
            return ts_tmm_fd
        else:
            return sam_tmm_td_, sam_tmm_fd_

    def cost(p, freq_idx_):
        n = np.array([1, p[0] + 1j * p[1], 1], dtype=complex)
        # n = array([1, 1.9+1j*0.1, p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[freq_idx_]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac) * phase_shift[freq_idx_]

        sam_tmm_fd = t_tmm_fd * ref_fd[freq_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(sam_fd[freq_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd) - phi[freq_idx_]) ** 2

        return amp_loss + phi_loss

    n_sub_a = sub_refidx_a(img_, point)

    res = None
    sigma, epsilon_r, n_opt = zero.copy(), zero.copy(), zero.copy()
    for f_idx_, freq in enumerate(freqs):
        if f_idx_ not in f_opt_idx:
            continue

        bounds_ = [(1.95, 2.00), (0.000, 0.010)]
        if freq > 1.7:
            bounds_ = [(1.965, 1.97), (0.007, 0.015)]
        n_sub_a_f_idx = n_sub_a[f_idx_, 1]
        bounds_ = [(n_sub_a_f_idx.real - 0.05, n_sub_a_f_idx.real + 0.05),
                   (n_sub_a_f_idx.imag - 0.005, n_sub_a_f_idx.imag + 0.005)]

        cost_ = cost
        if freq <= 0.150:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
        elif freq <= 2.0:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
            iters = initial_shgo_iters
            while res.fun > 1e-14:
                iters += 1
                res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=iters)
                if iters >= initial_shgo_iters + 3:
                    break
        else:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)

        n_opt[f_idx_] = res.x[0] + 1j * res.x[1]
        epsilon_r[f_idx_] = n_opt[f_idx_] ** 2
        sigma[f_idx_] = 1j * (1 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # "WORKS"
        # sigma[f_idx_] = 1j * (4 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # 1/(Ohm cm)
        # sigma[f_idx_] = 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        # sigma[f_idx_] = - 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        print(f"Substrate result: {np.round(sigma[f_idx_], 1)} (S/cm), "
              f"n: {np.round(n_opt[f_idx_], 3)}, at {np.round(freqs[f_idx_], 3)} THz, "
              f"loss: {res.fun}")

    n_opt[:f_opt_idx[0]] = n_opt[f_opt_idx[0]] * np.ones_like(n_opt[:f_opt_idx[0]])
    n_opt[f_opt_idx[-1]:] = n_opt[f_opt_idx[-1]] * np.ones_like(n_opt[f_opt_idx[-1]:])

    if en_plot:
        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_opt)
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1])))
        plt.figure("Spectrum substrate")
        plt.title("Spectrum substrate")
        plt.plot(ref_fd[plot_range1, 0], to_db(ref_fd[plot_range1, 1]) - noise_floor, label="Reference")
        plt.plot(sam_fd[plot_range1, 0], to_db(sam_fd[plot_range1, 1]) - noise_floor, label="Substrate measurement")
        plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1]) - noise_floor,
                 label="TMM fit")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

        t_tmm = calc_model(n_opt, ret_t=True)
        phi_tmm = np.angle(t_tmm)
        plt.figure("Phase substrate")
        plt.title("Phases substrate")
        plt.plot(freqs[plot_range1], phi[plot_range1], label="Substrate measurement", linewidth=2)
        plt.plot(freqs[plot_range1], phi_tmm[plot_range1], label="TMM", linewidth=2)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

        plt.figure("Time domain substrate")
        plt.plot(ref_td[:, 0], ref_td[:, 1], label="Ref Meas substrate", linewidth=2)
        plt.plot(sam_td[:, 0], sam_td[:, 1], label="Sam Meas substrate", linewidth=4)
        plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], linewidth=2, label="TMM")

        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (arb. u.)")

    # n_opt.real[:-29] = np.convolve(n_opt.real, np.ones(30) / 30, mode='valid')
    # n_opt.imag[:-29] = np.convolve(n_opt.imag, np.ones(30) / 30, mode='valid')

    t_abs_meas = np.abs(sam_fd[:, 1] / ref_fd[:, 1])
    T, R = calc_model(n_opt, ret_T_and_R=True)
    t_abs = np.sqrt(T)
    t_abs = np.abs(calc_model(n_opt, ret_t=True))

    sigma_ret = sigma[f_opt_idx]
    epsilon_r_ret = epsilon_r[f_opt_idx]
    n = np.array([freqs, n_opt], dtype=complex).T
    t_abs = t_abs[f_opt_idx]
    R = R[f_opt_idx]
    t_abs_meas = t_abs_meas[f_opt_idx]

    ret = {"loss": res.fun, "sigma": sigma_ret, "epsilon_r": epsilon_r_ret, "n": n,
           "t_abs": t_abs, "R": R, "t_abs_meas": t_abs_meas, "sam_fd": sam_fd, "ref_fd": ref_fd}

    return ret


def conductivity(img_, measurement_, d_film_=None):
    en_plot_ = True
    sub_point = (49, -5)
    sub_point = (22, -5)

    if "sample3" in str(img_.data_path):
        d_film = 0.350
    elif "sample4" in str(img_.data_path):
        d_film = 0.250
    else:
        d_film = d_film_

    ret_sub_eval = sub_refidx_tmm(img_, point=sub_point)
    if isinstance(ret_sub_eval, dict):
        n_sub = ret_sub_eval["n"]
    else:
        n_sub = ret_sub_eval

    # n_sub = np.array([n_sub[:, 0].real, (1.948+0.014j) * np.ones_like(n_sub[:, 1])]).T
    # return n_sub
    # n_sub[:, 1].real = 1.95*np.ones_like(n_sub[:, 1]).real

    shgo_bounds = [(1, 100), (1, 100)]

    if isinstance(measurement_, tuple):
        measurement_ = img_.get_measurement(*measurement_)
    print("Measurement file:", measurement_.filepath)
    film_td = measurement_.get_data_td()
    meas_pos = measurement_.position
    film_ref_td = img_.get_ref(both=False, point=(meas_pos[0], meas_pos[1]))

    # film_td = window(film_td, win_len=40, shift=0, en_plot=False, slope=1)
    # film_ref_td = window(film_ref_td, win_len=40, shift=0, en_plot=False, slope=1)

    film_td[:, 0] -= film_td[0, 0]
    film_ref_td[:, 0] -= film_ref_td[0, 0]

    film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

    # film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.5, 1.6), ret_fd=True, en_plot=False, extrapolate=True)
    # film_fd = phase_correction(film_fd, fit_range=(0.5, 1.6), ret_fd=True, en_plot=False, extrapolate=True)

    # phi = self.get_phase(point)
    from teval.functions import unwrap
    phi = np.angle(film_fd[:, 1] / film_ref_fd[:, 1])

    freqs = film_ref_fd[:, 0].real
    zero = np.zeros_like(freqs, dtype=complex)
    one = np.ones_like(freqs, dtype=complex)
    omega = 2 * pi * freqs

    f_opt_idx = f_axis_idx_map(freqs, freq_range)

    d_list = np.array([np.inf, d_sub, d_film, np.inf], dtype=float)

    phase_shift = np.exp(-1j * (d_sub + np.sum(d_film)) * omega / c_thz)

    # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

    def tinkham_approx():
        sub_sam_fd, sub_ref_fd = ret_sub_eval["sam_fd"], ret_sub_eval["ref_fd"]

        T_sub = sub_sam_fd[:, 1] / sub_ref_fd[:, 1]
        T_sam = film_fd[:, 1] / film_ref_fd[:, 1]

        sigma_tink = 0.01 * (1 + n_sub[:, 1]) * epsilon_0 * c0 * (T_sub - T_sam) / (T_sam * d_film * 1e-6)
        sigma_tink = 1 / (sigma_tink * d_film * 1e-4)

        return np.array([freqs, sigma_tink], dtype=complex).T

    def calc_model(n_model, ret_t=False, ret_T_and_R=False):
        n_list_ = np.array([one, n_sub[:, 1], n_model, one], dtype=complex).T

        R = np.zeros_like(freqs, dtype=complex)
        T = np.zeros_like(freqs, dtype=complex)
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx_, freq_ in enumerate(freqs):
            if np.isclose(freq_, 0):
                continue
            lam_vac = c_thz / freq_
            n = n_list_[f_idx_]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac) * phase_shift[f_idx_]
            ts_tmm_fd[f_idx_] = t_tmm_fd
            if ret_T_and_R:
                dict_res = coh_tmm_full("s", n, d_list, angle_in, lam_vac)
                T[f_idx_] = dict_res["T"]
                R[f_idx_] = dict_res["R"]

        sam_tmm_fd_ = np.array([freqs, ts_tmm_fd * film_ref_fd[:, 1]]).T
        sam_tmm_td_ = do_ifft(sam_tmm_fd_)
        sam_tmm_td_[:, 0] -= sam_tmm_td_[0, 0]

        if ret_T_and_R:
            return T, R
        if ret_t:
            return ts_tmm_fd
        else:
            return sam_tmm_td_, sam_tmm_fd_

    def cost(p, freq_idx_):
        n = np.array([1, n_sub[freq_idx_, 1], p[0] + 1j * p[1], 1], dtype=complex)
        # n = array([1, 1.9+1j*0.1, p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[freq_idx_]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac) * phase_shift[freq_idx_]

        sam_tmm_fd = t_tmm_fd * film_ref_fd[freq_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[freq_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd) - phi[freq_idx_]) ** 2

        return amp_loss + phi_loss

    res = None
    sigma, epsilon_r, n_opt = zero.copy(), zero.copy(), zero.copy()
    for f_idx_, freq in enumerate(freqs):
        if f_idx_ not in f_opt_idx:
            continue

        bounds_ = shgo_bounds

        cost_ = cost
        if freq <= 0.150:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
        elif freq <= 2.0:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)
            iters = initial_shgo_iters
            while res.fun > 1e-14:
                iters += 1
                res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=iters)
                if iters >= initial_shgo_iters + 3:
                    break
        else:
            res = shgo(cost_, bounds=bounds_, args=(f_idx_,), iters=initial_shgo_iters)

        n_opt[f_idx_] = res.x[0] + 1j * res.x[1]
        epsilon_r[f_idx_] = n_opt[f_idx_] ** 2
        sigma[f_idx_] = 1j * (1 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # "WORKS"
        # sigma[f_idx_] = 1j * (4 - epsilon_r[f_idx_]) * epsilon_0 * omega[f_idx_] * THz * 0.01  # 1/(Ohm cm)
        # sigma[f_idx_] = 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        # sigma[f_idx_] = - 1j * epsilon_r[f_idx_] * epsilon_0 * omega[f_idx_] * THz
        print(f"Result: {np.round(sigma[f_idx_], 1)} (S/cm), "
              f"n: {np.round(n_opt[f_idx_], 3)}, at {np.round(freqs[f_idx_], 3)} THz, "
              f"loss: {res.fun}")
        print(f"Substrate refractive index: {np.round(n_sub[f_idx_, 1], 3)}\n")
    n_opt[:f_opt_idx[0]] = n_opt[f_opt_idx[0]] * np.ones_like(n_opt[:f_opt_idx[0]])
    n_opt[f_opt_idx[-1]:] = n_opt[f_opt_idx[-1]] * np.ones_like(n_opt[f_opt_idx[-1]:])

    if en_plot_:
        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_opt)
        noise_floor = np.mean(20 * np.log10(np.abs(film_ref_fd[film_ref_fd[:, 0] > 6.0, 1])))
        plt.figure("Spectrum coated")
        plt.title("Spectrum coated")
        plt.plot(film_ref_fd[plot_range1, 0], to_db(film_ref_fd[plot_range1, 1]) - noise_floor, label="Reference")
        plt.plot(film_fd[plot_range1, 0], to_db(film_fd[plot_range1, 1]) - noise_floor, label="Coated")
        plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1]) - noise_floor,
                 label="TMM fit")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")

        t_tmm = calc_model(n_opt, ret_t=True)
        phi_tmm = np.angle(t_tmm)

        plt.figure("Phase coated")
        plt.title("Phases coated")
        plt.plot(freqs[plot_range1], phi[plot_range1], label="Measured", linewidth=2)
        plt.plot(freqs[plot_range1], phi_tmm[plot_range1], label="TMM", linewidth=2)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase difference (rad)")

        plt.figure("Time domain")
        plt.plot(film_ref_td[:, 0], film_ref_td[:, 1], label="Ref Meas", linewidth=2)
        plt.plot(film_td[:, 0], film_td[:, 1], label="Sam Meas", linewidth=4)
        plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], linewidth=2, label="TMM")

        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (arb. u.)")

    t_abs_meas = np.abs(film_fd[:, 1] / film_ref_fd[:, 1])
    T, R = calc_model(n_opt, ret_T_and_R=True)
    t_abs = np.sqrt(T)
    t_abs = np.abs(calc_model(n_opt, ret_t=True))

    if len(f_opt_idx) != 1:
        sigma_ret = np.array([freqs[f_opt_idx], sigma[f_opt_idx]], dtype=complex).T
        epsilon_r_ret = np.array([freqs[f_opt_idx], epsilon_r[f_opt_idx]], dtype=complex).T
        n = np.array([freqs[f_opt_idx], n_opt[f_opt_idx]], dtype=complex).T
        t_abs = np.array([freqs[f_opt_idx], t_abs[f_opt_idx]], dtype=complex).T
        R = np.array([freqs[f_opt_idx], R[f_opt_idx]], dtype=complex).T
        t_abs_meas = np.array([freqs[f_opt_idx], t_abs_meas[f_opt_idx]], dtype=complex).T
    else:
        sigma_ret = sigma[f_opt_idx]
        epsilon_r_ret = epsilon_r[f_opt_idx]
        n = n_opt[f_opt_idx]
        t_abs = t_abs[f_opt_idx]
        R = R[f_opt_idx]
        t_abs_meas = t_abs_meas[f_opt_idx]

    eval_res = {"loss": res.fun, "sigma": sigma_ret, "epsilon_r": epsilon_r_ret, "n": n,
                "t_abs": t_abs, "R": R, "t_abs_meas": t_abs_meas, "tinkham": tinkham_approx()}

    return eval_res
