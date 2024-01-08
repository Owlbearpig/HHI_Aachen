import numpy as np
from numpy import pi
from scipy.constants import epsilon_0
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map, do_ifft, to_db
from teval.consts import c_thz, THz, plot_range1
from tmm import coh_tmm as coh_tmm_full
from tmm_slim import coh_tmm
import matplotlib.pyplot as plt
from scipy.optimize import shgo


d_sub = 1000
freq_range = (0.25, 2.5)
angle_in = 0.0
initial_shgo_iters = 3


def sub_refidx(img_, point=(22.5, 5)):
    #img_.plot_point(*point)
    #plt.show()
    sub_meas = img_.get_measurement(*point)
    sam_td = sub_meas.get_data_td()
    ref_td = img_.get_ref(point=point)

    sam_td = window(sam_td, en_plot=False)
    ref_td = window(ref_td, en_plot=False)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

    freqs = ref_fd[:, 0].real
    omega = 2*np.pi*freqs

    phi_ref = phase_correction(ref_fd)
    phi_sam = phase_correction(sam_fd)
    phi_diff = phi_sam[:, 1] - phi_ref[:, 1]

    n0 = 1 + c_thz * phi_diff / (omega * d_sub)
    k0 = -c_thz * np.log(np.abs(sam_fd[:, 1]/ref_fd[:, 1]) * (1+n0)**2 / (4*n0)) / (omega*d_sub)

    return np.array([freqs, n0+1j*k0], dtype=complex).T


def conductivity(img_, measurement_, d_film_=None):
    en_plot_ = True
    sub_point = (22, -4)

    if "sample3" in str(img_.data_path):
        d_film = 0.350
    elif "sample4" in str(img_.data_path):
        d_film = 0.250
    else:
        d_film = d_film_

    n_sub = sub_refidx(img_, point=sub_point)
    n_sub[:, 1] = 1.99*np.ones_like(n_sub[:, 1]) + 1j*0.013 * np.ones_like(n_sub[:, 1])

    shgo_bounds = [(1, 100), (1, 100)]

    if isinstance(measurement_, tuple):
        measurement_ = img_.get_measurement(*measurement_)

    film_td = measurement_.get_data_td()
    film_ref_td = img_.get_ref(both=False, point=measurement_.position)

    film_td = window(film_td, win_len=16, shift=0, en_plot=False, slope=0.99)
    film_ref_td = window(film_ref_td, win_len=16, shift=0, en_plot=False, slope=0.99)

    film_td[:, 0] -= film_td[0, 0]
    film_ref_td[:, 0] -= film_ref_td[0, 0]

    film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

    # film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.1, 0.2), ret_fd=True, en_plot=False)
    # film_fd = phase_correction(film_fd, fit_range=(0.1, 0.2), ret_fd=True, en_plot=False)

    # phi = self.get_phase(point)
    phi = np.angle(film_fd[:, 1] / film_ref_fd[:, 1])

    freqs = film_ref_fd[:, 0].real
    zero = np.zeros_like(freqs, dtype=complex)
    one = np.ones_like(freqs, dtype=complex)
    omega = 2 * pi * freqs

    f_opt_idx = f_axis_idx_map(freqs, freq_range)

    d_list = np.array([np.inf, d_sub, d_film, np.inf], dtype=float)

    phase_shift = np.exp(-1j * (d_sub + np.sum(d_film)) * omega / c_thz)

    # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

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

    ret = {"loss": res.fun, "sigma": sigma_ret, "epsilon_r": epsilon_r_ret, "n": n,
           "t_abs": t_abs, "R": R, "t_abs_meas": t_abs_meas}

    return ret
