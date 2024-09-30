import logging

import numpy as np
from numpy import pi
from scipy.constants import epsilon_0
from teval.functions import phase_correction, window, do_fft, f_axis_idx_map, do_ifft, to_db
from teval.consts import c_thz, THz, plot_range1, c0
from tmm import coh_tmm as coh_tmm_full
from tmm_slim import coh_tmm
import matplotlib.pyplot as plt
from scipy.optimize import shgo
from functools import partial
from enum import Enum
from teval import Image

options = {"d_sub": 1000, "freq_range": (0.25, 2.5),
           "angle_in": 0.0, "init_shgo_iters": 3}


class Method(Enum):
    pass


class ConductivityEval:
    options = {}
    def __init__(self, film_img_path, sub_img_path=None, options_dict=None):
        self.film_img = Image(film_img_path)

        if sub_img_path is None:
            sub_img_path = film_img_path

        self.sub_img = Image(sub_img_path)

        self._set_options(options_dict)

    def _set_options(self, options_=None):
        if options_ is None:
            options_ = {}

        default_options = {"d_sub": 1000,
                           "freq_range": (0.25, 2.5),
                           "angle_in": 0.0,
                           "init_shgo_iters": 3
                           }

        for k in default_options:
            if not k in options_:
                logging.warning(f"Setting {k} to default value {default_options[k]}")
                options_[k] = default_options[k]

        self.options.update(default_options)

    def sub_refidx_a(self, point):
        #img_.plot_point(*point)
        #plt.show()
        d = self.options["d_sub"]
        res = self.sub_img.evaluate_point(point, d)

        return res["n"]

    def sub_refidx_tmm(self, point, selected_freq_=None):
        initial_shgo_iters = self.options["init_shgo_iters"]
        freq_range = self.options["freq_range"]
        d_sub = self.options["d_sub"]
        angle_in = self.options["angle_in"]
        en_plot = False

        sam_meas = self.sub_img.get_measurement(*point)
        ref_meas = self.sub_img.find_nearest_ref(sam_meas)

        ref_td, sam_td = ref_meas.get_data_td(), sam_meas.get_data_td()

        sam_td[:, 0] -= sam_td[0, 0]

        # sam_td = window(sam_td, en_plot=False, win_len=10)
        # ref_td = window(ref_td, en_plot=False, win_len=10)

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)
        phi = np.angle(sam_fd[:, 1] / ref_fd[:, 1])

        freqs = ref_fd[:, 0].real

        zero = np.zeros_like(freqs, dtype=complex)
        one = np.ones_like(freqs, dtype=complex)
        omega = 2 * np.pi * freqs

        if selected_freq_ is None:
            f_opt_idx = f_axis_idx_map(freqs, freq_range)
        else:
            f_opt_idx = list(f_axis_idx_map(freqs, selected_freq_))

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
                t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[f_idx_]
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
            t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[freq_idx_]

            sam_tmm_fd = t_tmm_fd * ref_fd[freq_idx_, 1]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(sam_fd[freq_idx_, 1])) ** 2
            phi_loss = (np.angle(t_tmm_fd) - phi[freq_idx_]) ** 2

            return amp_loss + phi_loss

        n_sub_a = self.sub_refidx_a(point)

        res = None
        sigma, epsilon_r, n_opt = zero.copy(), zero.copy(), zero.copy()
        for f_idx_, freq in enumerate(freqs):
            if f_idx_ not in f_opt_idx:
                continue

            bounds_ = [(1.95, 2.00), (0.000, 0.010)]
            if freq > 1.7:
                bounds_ = [(1.965, 1.99), (0.007, 0.015)]
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

        #n_opt.real[:-29] = np.convolve(n_opt.real, np.ones(30) / 30, mode='valid')
        #n_opt.imag[:-29] = np.convolve(n_opt.imag, np.ones(30) / 30, mode='valid')

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

        eval_ret = {"n": n, "sam_fd": sam_fd, "ref_fd": ref_fd, "ref_td": ref_td, "sam_td": sam_td}

        return eval_ret

    def conductivity(self, img_, measurement_, d_film_=None, selected_freq_=2.000, ref_idx=0,
                     tinkham_method=False, shift_sub=False, p2p_method=False):
        initial_shgo_iters = 3
        d_sub = self.options["d_sub"]
        angle_in = self.options["angle_in"]
        # sub_point = (49, -5)
        sub_point = (22, -5)
        sub_point = (40, -10)

        if shift_sub:
            from itertools import product
            from random import choice
            sub_points = list(product(range(21, 24), range(-14, 10)))
            sub_points.extend(list(product(range(46, 50), range(-14, 10))))
            sub_point = choice(sub_points)

            sub_point = (49, -5)
            sub_point = (22, -5)
            sub_point = (40, -10)

            print(f"New sub point: {[sub_point[0], sub_point[1]]}")

        if "sample3" in str(img_.data_path):
            d_film = 0.350
        elif "sample4" in str(img_.data_path):
            d_film = 0.250
        else:
            d_film = d_film_

        sub_eval_res = self.sub_refidx_tmm(img_, point=sub_point, selected_freq_=selected_freq_)
        n_sub = sub_eval_res["n"]
        sub_ref_fd = sub_eval_res["ref_fd"]
        sub_sam_fd = sub_eval_res["sam_fd"]
        sub_ref_td = sub_eval_res["ref_td"]
        sub_sam_td = sub_eval_res["sam_td"]

        shgo_bounds = [(1, 75), (1, 75)]

        if isinstance(measurement_, tuple):
            measurement_ = img_.get_measurement(*measurement_)

        film_td = measurement_.get_data_td()
        meas_pos = measurement_.position

        shifts = [[-4, 0], [-2, 0], [2, 0], [4, 0]]
        if ref_idx:
            try:
                shift = shifts[ref_idx]
            except IndexError:
                shift = [ref_idx, 0]
            meas_pos = (meas_pos[0] + shift[0], meas_pos[1] + shift[1])

        film_ref_td = img_.get_ref(both=False, point=meas_pos)
        # film_ref_td = img_.get_ref(both=False)

        # film_td = window(film_td, win_len=16, shift=0, en_plot=False, slope=0.99)
        # film_ref_td = window(film_ref_td, win_len=16, shift=0, en_plot=False, slope=0.99)

        pos_x = (measurement_.position[0] < 25) or (45 < measurement_.position[0])
        pos_y = (measurement_.position[1] < -11) or (9 < measurement_.position[1])
        if (np.max(film_td[:, 1]) / np.max(film_ref_td[:, 1]) > 0.50) or (pos_x and pos_y):
            return 1000, 0

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

        f_opt_idx = f_axis_idx_map(freqs, selected_freq_)

        d_list = np.array([np.inf, d_sub, d_film, np.inf], dtype=float)

        phase_shift = np.exp(-1j * (d_sub + np.sum(d_film)) * omega / c_thz)

        # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

        def tinkham_approx():
            T_sub = sub_sam_fd[:, 1] / sub_ref_fd[:, 1]
            T_sam = film_fd[:, 1] / film_ref_fd[:, 1]

            sigma_tink = 0.01 * (1 + n_sub[:, 1]) * epsilon_0 * c0 * (T_sub - T_sam) / (T_sam * d_film * 1e-6)
            sigma_tink = 1 / (sigma_tink * d_film * 1e-4)

            return np.array([sub_ref_fd[:, 0], sigma_tink], dtype=complex).T

        if tinkham_method:
            sigma = tinkham_approx()

            return 1 / (sigma[f_opt_idx[0], 1].real * d_film * 1e-4)

        def p2p_approx():
            T_sub = np.max(sub_sam_td[:, 1].real) / np.max(sub_ref_td[:, 1].real)
            T_sam = np.max(film_td[:, 1]) / np.max(film_ref_td[:, 1])

            sigma_tink = 0.01 * (1 + n_sub[:, 1]) * epsilon_0 * c0 * (T_sub - T_sam) / (T_sam * d_film * 1e-6)
            sigma_tink = 1 / (sigma_tink * d_film * 1e-4)

            return np.array([sub_ref_fd[:, 0], sigma_tink], dtype=complex).T

        if p2p_method:
            sigma = p2p_approx()

            return 0.01 / (sigma[f_opt_idx[0], 1].real * d_film * 1e-4)

        def cost(p, freq_idx_):
            n = np.array([1, n_sub[freq_idx_, 1], p[0] + 1j * p[1], 1], dtype=complex)
            # n = array([1, 1.9+1j*0.1, p[0] + 1j * p[1], 1])
            lam_vac = c_thz / freqs[freq_idx_]
            t_tmm_fd = coh_tmm_full("s", n, d_list, angle_in, lam_vac)["t"] * phase_shift[freq_idx_]

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

        # mean_cond = 0.5*(sigma[f_opt_idx[0]].real + sigma[f_opt_idx[0]].imag)

        return 1 / (sigma[f_opt_idx[0]].real * d_film * 1e-4), res.fun

    def conductivity_impl(self, measurement, selected_freq):
        sheet_resistance, err = self.conductivity(measurement, selected_freq_=selected_freq)

        retries = 1
        while (err > 1e-10) and (retries > 0):
            print(f"Shifting ref. point (min. func. val: {err})")
            sheet_resistance, err = self.conductivity(measurement, selected_freq_=selected_freq, shift_sub=True)
            retries -= 1

        return sheet_resistance


if __name__ == '__main__':
    # sub_img_p =
    new_eval = ConductivityEval()