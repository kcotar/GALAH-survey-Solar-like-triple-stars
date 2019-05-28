import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import joblib
from lmfit.models import SkewedGaussianModel
import emcee, corner, imp
import thecannon as tc
from astropy.table import Table, join
from time import time
from copy import deepcopy
from os import path
from socket import gethostname
np.warnings.filterwarnings('ignore')

# PC hostname
pc_name = gethostname()
USE_IDR3 = True

dr53_dir = '/shared/ebla/cotar/dr5.3/'
data_dir = '/shared/ebla/cotar/'
out_dir_root = '/shared/data-camelot/cotar/_Multiples_binaries_results_iDR3/'

imp.load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
from helper_functions import *

# read galah, gaia, cannon, and photometric data for all galah objects
if USE_IDR3:
    galah_gaia_data = Table.read(data_dir + 'galah_cannon_DR3_gaia_photometry_20181221_ebv-corr.fits')
else:
    galah_gaia_data = Table.read(data_dir + 'galah_cannon_gaia_photometry_20180327_ebv-corr.fits')

# remove unused columns
galah_gaia_data.remove_columns(['gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'ymag', 'e_ymag'])
# read precomputed photometry table
if USE_IDR3:
    median_photometry_data = Table.read(data_dir + 'median_photometry_cannon0_DR3_ruwe_80_005_01_all_ebv.fits')  # with or without parallax filtering
else:
    median_photometry_data = Table.read(data_dir + 'median_photometry_cannon0_ruwe_80_005_01_all_ebv.fits')  # with or without parallax filtering
# median_photometry_data.remove_columns(['gmag', 'rmag', 'imag', 'zmag', 'ymag'])
# read t-SNE classes determined by Gregor
galah_tsne = Table.read(data_dir + 'tsne_class_1_0.csv')
bin_sid = galah_tsne[galah_tsne['published_reduced_class_proj1'] == 'binary']['sobject_id']
# remove SB2 binaries from the galah set
idx_bin = np.in1d(galah_gaia_data['sobject_id'], bin_sid)
idx_ok = np.logical_not(idx_bin)
idx_parallax_ok = galah_gaia_data['ruwe'] <= 1.4
idx_ok = np.logical_and(idx_ok, idx_parallax_ok)
galah_tsne = None

# Canonn linelist
galah_linelist = Table.read(data_dir + 'GALAH_Cannon_linelist_newer.csv')

# preload and prepare everything - drugace se pojavlja nek cuden error ce to delam znotraj sample procedure
# load cannon spectral model created by Gregor
if USE_IDR3:
    cannon_model = tc.CannonModel.read(data_dir + 'model_cannon181221_DR3_ccd1234_noflat_red0_cannon0_oksnr_vsiniparam_dwarfs.dat')
else:
    cannon_model = tc.CannonModel.read(data_dir + 'model_cannon180325_ccd1234_noflat_red0_cannon0_oksnr_vsiniparam.dat')

thetas = cannon_model.theta
vectorizer = cannon_model.vectorizer
fid = cannon_model._fiducials
sca = cannon_model._scales
print cannon_model.dispersion

# # load cannon photometric model created by me
# cannon_model_p = tc.CannonModel.read('model_cannon180325_photometry_cannon0_parallaxok.dat')
# cannon_txt = open('model_cannon180325_photometry_cannon0_parallaxok_cols.txt', 'r')
# cannon_model_p_cols = cannon_txt.read().split(',')
# cannon_txt.close()
# thetas_p = cannon_model_p.theta
# vectorizer_p = cannon_model_p.vectorizer
# fid_p = cannon_model_p._fiducials
# sca_p = cannon_model_p._scales

p_cols = ['Bmag','Vmag','gpmag','rpmag','ipmag','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','Jmag','Hmag','Kmag','W1mag','W2mag']#, 'gmag', 'rmag', 'imag', 'zmag', 'ymag']#,'W3mag','W4mag']
p_cols_galah = ['Bmag','Vmag','rpmag','ipmag']
# p_cols = ['Bmag','Vmag','gpmag','rpmag','ipmag','Jmag','Hmag','Kmag','W1mag','W2mag']#,'W3mag','W4mag']
p_cols_sigma = ['e_'+c for c in p_cols]


def get_sobjects_from_final_selection(full_path):
    sel_txt = open(full_path, 'r')
    solar_like_sobjects = sel_txt.read()
    sel_txt.close()
    solar_like_sobjects = [np.int64(sid) for sid in solar_like_sobjects.split(',')]
    return np.array(solar_like_sobjects)


def get_spectra_complete(s_id, get_bands=[1,2,3,4]):
    flx, wvl, sig = get_spectra_dr52(str(s_id), bands=get_bands, root=dr53_dir, extension=4, read_sigma=True)
    # TODO: if needed read extension 0 and normalize spectrum with my procedure, same as in solar twin search,
    # procedure is not consistent with used Cannon model
    return np.hstack(flx), np.hstack(sig)*np.hstack(flx), np.hstack(wvl)


def get_linelist_mask(wvl_values, d_wvl=0., element=None):
    idx_lines_mask = wvl_values < 0.

    if element is None:
        galah_linelist_use = deepcopy(galah_linelist)
    else:
        galah_linelist_use = galah_linelist[galah_linelist['Element'] == element]

    for line in galah_linelist_use:
        idx_lines_mask[np.logical_and(wvl_values >= line['line_start'] - d_wvl, wvl_values <= line['line_end'] + d_wvl)] = True

    return idx_lines_mask


def plot_lnprob(chain_vals, fit_sel_vals, path, write_out=True):
    c_fig = corner.corner(chain_vals, truths=fit_sel_vals,
                          quantiles=[0.16, 0.5, 0.84], 
                          show_titles=True, range=[(5600, 6100), (5100, 5900), (4750, 5650)], title_fmt='.0f',
                          labels=[u'T$_{eff1}$ [K]', u'T$_{eff2}$ [K]', u'T$_{eff3}$ [K]'], bins=60, plot_contours=False)
    if write_out:
        # c_fig.tight_layout()
        c_fig.subplots_adjust(left=0.105, bottom=0.105)
        c_fig.savefig(path, dpi=250)
    plt.close(c_fig)


def plot_walkers(walkers_prob, path, write_out=True, miny_perc=18., sigma_lnprob=None):
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(7, 5.))
    for i_w in range(walkers_prob.shape[0]):
        plt.plot(walkers_prob[i_w, :], lw=0.2)
    walkers_last = walkers_prob[:, 20].flatten()
    walkers_prob = walkers_prob.flatten()  # without this correction numpy
    walkers_prob = walkers_prob[np.isfinite(walkers_prob)]  # percentile may return incorrect -inf value
    if len(walkers_prob) < 5:
        return
    if sigma_lnprob is None:
        min_prob = np.nanpercentile(walkers_prob, 50.)
        sigma_lnprob = np.nanstd(walkers_last[walkers_last > min_prob])
    # plt.title('Lnprob best 1%: {:.1f}, sigma selection: {:.1f}'.format(np.nanpercentile(walkers_prob, 99), sigma_lnprob))
    # plt.ylim((np.nanpercentile(walkers_prob, miny_perc), np.nanpercentile(walkers_prob, 99.9)))
    plt.ylim((350, np.nanpercentile(walkers_prob, 100.)+5))
    # plt.axhline(np.nanpercentile(walkers_prob, 25.), color='black', ls='--', lw=1)
    # plt.axhline(np.nanpercentile(walkers_prob, 50.), color='black', ls='--', lw=1)
    # plt.axhline(np.nanpercentile(walkers_prob, 75.), color='black', ls='--', lw=1)
    plt.xlim(-3, 203)
    plt.grid(ls='--', alpha=0.2, color='black')
    plt.ylabel('Log-probability value')
    plt.xlabel('Sequential number of the walkers step')
    if write_out:
        plt.tight_layout()
        plt.savefig(path, dpi=300)
    plt.close()
    plt.rcParams['font.size'] = 12


def get_distribution_peaks(chain_vals, plot_ref='', plot=False, write_out=True):
    n_par = chain_vals.shape[1]
    peak_center = list([])
    for i_p in range(n_par):
        data = chain_vals[:, i_p]
        hist, bins = np.histogram(data, range=(np.percentile(data, 2.), np.percentile(data, 98.)), bins=150)
        d_bin = bins[1] - bins[0]
        bins = bins[:-1] + d_bin / 2.
        model = SkewedGaussianModel()
        params = model.make_params(amplitude=np.max(hist), center=np.nanmedian(data), sigma=1, gamma=0)
        result = model.fit(hist, params, x=bins)
        # print result.fit_report()
        hist_peak = bins[np.argmax(result.best_fit)]
        peak_center.append(hist_peak)
        if plot:
            # output results
            plt.plot(bins, hist, c='black')
            plt.axvline(x=hist_peak)
            plt.plot(bins, result.best_fit, c='red')
            if write_out:
                plt.savefig(plot_ref + '_chainvals_' + str(i_p) + '.png', dpi=250)
            plt.close()
    return peak_center


def _prepare_hist_data(d, bins, range, norm=True):
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.nanmax(heights)
    return edges[:-1], heights, width


def fit_skewed(edges, hist):
    bins = edges + (edges[1]-edges[0])/2.
    model = SkewedGaussianModel()
    params = model.make_params(amplitude=np.max(hist), center=bins[np.argmax(hist)], sigma=1., gamma=0.)
    result = model.fit(hist, params, x=bins)
    hist_peak = bins[np.argmax(result.best_fit)]
    return hist_peak, bins, result.best_fit


def median_photometry(teff, logg, feh, p_cols,
                      d_teff=50, d_logg=0.25, d_feh=0.1, plot_res=False, path=None, min_data=20, idx_init=None):
    idx_use = galah_gaia_data['flag_cannon'] == 0
    # idx_use = np.logical_and(data['flag_cannon'] == 0, data['red_flag'] == 0)
    if idx_init is not None:
        idx_use = np.logical_and(idx_use, idx_init)
    # idx_use = np.logical_and(idx_use, np.isfinite(galah_gaia_data['parallax']))
    idx_use = np.logical_and(idx_use, np.isfinite(galah_gaia_data['r_est']))
    idx_use = np.logical_and(idx_use, np.logical_and(galah_gaia_data['Teff_cannon'] >= teff-d_teff/2., galah_gaia_data['Teff_cannon'] <= teff+d_teff/2.))
    idx_use = np.logical_and(idx_use, np.logical_and(galah_gaia_data['Fe_H_cannon'] >= feh-d_feh/2., galah_gaia_data['Fe_H_cannon'] <= feh+d_feh/2.))
    idx_use = np.logical_and(idx_use, np.logical_and(galah_gaia_data['Logg_cannon'] >= logg-d_logg/2., galah_gaia_data['Logg_cannon'] <= logg+d_logg/2.))

    # get data
    data_use = galah_gaia_data[idx_use][p_cols]
    # print ','.join([str(sid) for sid in galah_gaia_data['sobject_id'][idx_use]])
    if len(data_use) < min_data:
        return np.full(len(p_cols), np.nan)
    # print ' Data points:', len(data_use), 'for', teff, logg, feh

    # print galah_gaia_data[idx_use]['Teff_cannon','Fe_H_cannon','Logg_cannon','e_Logg_cannon']
    # data statistics
    # if len(p_cols) > 1:
    # abs_mag_vals = data_use.to_pandas().values - 2.5*np.log10(((1e3/galah_gaia_data[idx_use]['parallax'])/10.)**2).reshape(-1, 1)
    abs_mag_vals = data_use.to_pandas().values - 2.5*np.log10(((galah_gaia_data[idx_use]['r_est'])/10.)**2).reshape(-1, 1)
    photo_med = np.nanmedian(abs_mag_vals, axis=0)
    # else:
    #     abs_mag_vals = data_use.data - 2.5*np.log10(((1e3/galah_gaia_data[idx_use]['parallax'])/10.)**2).reshape(-1, 1)
    #     photo_med = np.nanmedian(abs_mag_vals)

    if plot_res:
        n_x = 4
        n_y = 4
        fig, ax = plt.subplots(n_y, n_x, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.2, wspace=0.3, left=0.025, right=0.975, top=0.975, bottom=0.025)
        for i_p in range(len(p_cols)):
            x_p = i_p % n_x
            y_p = int(i_p / n_x)
            # plot_vals = data_use[p_cols[i_p]] - 2.5*np.log10(((1e3/galah_gaia_data[idx_use]['parallax'])/10.)**2)
            plot_vals = data_use[p_cols[i_p]] - 2.5*np.log10(((galah_gaia_data[idx_use]['r_est'])/10.)**2)
            plot_vals_median = np.nanmedian(plot_vals)
            plot_vals_std = 2. * np.nanstd(plot_vals)
            h_edg, h_hei, h_wid = _prepare_hist_data(plot_vals, 75, (np.nanpercentile(plot_vals, .25), np.nanpercentile(plot_vals, 99.75)))
            #
            plt_fit_val, x_fit, y_fit = fit_skewed(h_edg, h_hei)
            # plots
            ax[y_p, x_p].bar(h_edg, h_hei, width=h_wid, color='black', alpha=0.4)
            ax[y_p, x_p].axvline(x=plot_vals_median, color='red', ls='--', lw=1.5, alpha=0.75)
            ax[y_p, x_p].axvline(x=plot_vals_median - plot_vals_std, color='red', ls='--', lw=1.5, alpha=0.25)
            ax[y_p, x_p].axvline(x=plot_vals_median + plot_vals_std, color='red', ls='--', lw=1.5, alpha=0.25)
            ax[y_p, x_p].axvline(x=plt_fit_val, color='blue', ls='--', lw=1.5, alpha=0.75)
            ax[y_p, x_p].plot(x_fit, y_fit, color='blue', lw=1.5, alpha=0.75)
            ax[y_p, x_p].grid(ls='--', alpha=0.15, color='black')
            # write out labels to the plot
            ax[y_p, x_p].set(title=p_cols[i_p])
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)
        plt.close()

    return photo_med


def median_photometry_precomputed_intepol(teff, logg, feh, p_cols,
                                          d_teff=50, d_logg=0.25, d_feh=0.1):
    # select upper and lower bin for given values
    u_teff = np.unique(median_photometry_data['teff'])
    teff_sel = np.sort(u_teff[np.argsort(np.abs(u_teff - teff))[:2]])
    # teff_best = u_teff[np.argmin(np.abs(u_teff - teff))]
    u_logg = np.unique(median_photometry_data['logg'])
    logg_sel = np.sort(u_logg[np.argsort(np.abs(u_logg - logg))[:2]])
    u_feh = np.unique(median_photometry_data['feh'])
    # feh_sel = np.sort(u_feh[np.argsort(np.abs(u_feh - feh))[:2]])
    feh_best = u_feh[np.argmin(np.abs(u_feh - feh))]
    try:
        # get photometry values for both border values
        idx_feh_sel = median_photometry_data['feh'] == feh_best
        idx_00 = np.where(np.logical_and(idx_feh_sel,
                                          np.logical_and(median_photometry_data['teff'] == teff_sel[0],
                                                         median_photometry_data['logg'] == logg_sel[0])))[0]
        idx_01 = np.where(np.logical_and(idx_feh_sel,
                                         np.logical_and(median_photometry_data['teff'] == teff_sel[0],
                                                        median_photometry_data['logg'] == logg_sel[1])))[0]
        idx_10 = np.where(np.logical_and(idx_feh_sel,
                                         np.logical_and(median_photometry_data['teff'] == teff_sel[1],
                                                        median_photometry_data['logg'] == logg_sel[0])))[0]
        idx_11 = np.where(np.logical_and(idx_feh_sel,
                                         np.logical_and(median_photometry_data['teff'] == teff_sel[1],
                                                        median_photometry_data['logg'] == logg_sel[1])))[0]
        vals_00 = median_photometry_data[idx_00][p_cols].to_pandas().values
        vals_01 = median_photometry_data[idx_01][p_cols].to_pandas().values
        vals_10 = median_photometry_data[idx_10][p_cols].to_pandas().values
        vals_11 = median_photometry_data[idx_11][p_cols].to_pandas().values

        vals_low = vals_00 - (vals_00 - vals_10) * (teff_sel[0] - teff) / (teff_sel[0] - teff_sel[1])
        vals_hig = vals_01 - (vals_01 - vals_11) * (teff_sel[0] - teff) / (teff_sel[0] - teff_sel[1])

        final_vals = vals_low - (vals_low - vals_hig) * (logg_sel[0]-logg)/(logg_sel[0]-logg_sel[1])

        return Table(final_vals)
    except:
        return []


def median_photometry_precomputed(teff, logg, feh, p_cols,
                                  d_teff=50, d_logg=0.25, d_feh=0.1, get_bin_vals=False):
    # select best match based on given teff
    u_teff = np.unique(median_photometry_data['teff'])
    teff_sel = u_teff[np.argmin(np.abs(u_teff - teff))]
    if np.abs(teff_sel - teff) < d_teff / 2.:
        logg_vals = median_photometry_data['logg'][median_photometry_data['teff'] == teff_sel]
        u_logg = np.unique(logg_vals)
        logg_sel = u_logg[np.argmin(np.abs(u_logg - logg))]
        if np.abs(logg_sel - logg) < d_logg / 2.:
            feh_vals = median_photometry_data['feh'][np.logical_and(median_photometry_data['teff'] == teff_sel,
                                                                    median_photometry_data['logg'] == logg_sel)]
            u_feh = np.unique(feh_vals)
            feh_sel = u_feh[np.argmin(np.abs(u_feh - feh))]
            if np.abs(feh_sel - feh) < d_feh / 2.:
                idx_row = np.where(np.logical_and(median_photometry_data['feh'] == feh_sel,
                                                  np.logical_and(median_photometry_data['teff'] == teff_sel,
                                                                 median_photometry_data['logg'] == logg_sel)))[0]
                # print 'Precomputed:', teff, logg, feh, '  ---> ', median_photometry_data[idx_row]['teff', 'logg', 'feh'].to_pandas().values[0]
                if get_bin_vals:
                    return median_photometry_data[idx_row][p_cols], median_photometry_data[idx_row]['teff', 'logg', 'feh']
                else:
                    return median_photometry_data[idx_row][p_cols]
    if get_bin_vals:
        return [], []
    else:
        return []


# def median_photometry_cannon(teff, logg, feh, p_cols):
#     sint = thetas_p[:, 0] * 0.0
#     labs = (np.array([teff, logg, feh]) - fid_p) / sca_p
#     vec = vectorizer_p(labs)
#     for i, j in enumerate(vec):
#         sint += thetas_p[:, i] * j
#     return Table(sint, names=cannon_model_p_cols)[p_cols]


def eval_params(param, teff_range):
    if len(param) == 1:
        t1 = param
        t2 = teff_range[0] + 2
        t3 = teff_range[0] + 1
    elif len(param) == 2:
        t1, t2 = param
        t3 = teff_range[0] + 1
    else:
        t1, t2, t3 = param
    if not t1 >= t2:
        # print t1,t2,t3
        # print 'T1'
        return False
    if not t2 >= t3:
        # print 'T2'
        return False
    if not teff_range[0] < t1 < teff_range[1]:
        # print 'R1'
        return False
    if not teff_range[0] < t2 < teff_range[1]:
        # print 'R2'
        return False
    if not teff_range[0] < t3 < teff_range[1]:
        # print 'R3'
        return False
    return True


def _list_mag_photometry(teff_params, logg, feh, get_col):
    if len(logg) == 1:
        logg = np.repeat(logg, len(teff_params))
    # compute median object photometry
    phot_return = list([])
    for i_t in range(len(teff_params)):
        try:
            # narrower ranges + filtering
            phot_obj = median_photometry_precomputed_intepol(teff_params[i_t], logg[i_t], feh, get_col).to_pandas().values[0]
            # phot_obj = median_photometry(teff_params[i_t], logg[i_t], feh, get_col, d_teff=80, d_logg=0.05, d_feh=0.1, idx_init=idx_ok)
        except:
            # broader ranges + no filtering
            # print '  Using broader ranges - _list_mag_photometry'
            # print '  - for inputs', teff_params[i_t], logg[i_t], feh
            phot_obj = median_photometry(teff_params[i_t], logg[i_t], feh, get_col, d_teff=120, d_logg=0.2, d_feh=0.2, idx_init=idx_ok)
        phot_return.append(phot_obj)
    return phot_return


def _comb_mag_photometry(teff_params, logg, feh, return_flux=False):
    if len(logg) == 1:
        logg = np.repeat(logg, len(teff_params))
    # compute median object photometry
    j_comb = 0
    for i_t in range(len(teff_params)):
        try:
            # narrower ranges + filtering
            phot = median_photometry(teff_params[i_t], logg[i_t], feh, p_cols, d_teff=80, d_logg=0.05, d_feh=0.1, idx_init=idx_ok)
        except:
            # try broader ranges and no filtering
            # print '  Using broader ranges - _comb_mag_photometry'
            phot = median_photometry(teff_params[i_t], logg[i_t], feh, p_cols, d_teff=100, d_logg=0.2, d_feh=0.2, idx_init=idx_ok)
        if len(phot) == 0 or not np.isfinite(phot).all():
            return []
        j_comb += 10 ** (-0.4 * phot)
    if return_flux:
        return j_comb
    else:
        return -2.5 * np.log10(j_comb)


def _comb_mag_photometry_precomputed(teff_params, logg, feh, return_flux=False):
    if len(logg) == 1:
        logg = np.repeat(logg, len(teff_params))
    # compute median object photometry
    j_comb = 0
    for i_t in range(len(teff_params)):
        phot = median_photometry_precomputed_intepol(teff_params[i_t], logg[i_t], feh, p_cols)
        if len(phot) == 0:
            return []
        j_comb += 10 ** (-0.4 * phot.to_pandas().values[0])
    if return_flux:
        return j_comb
    else:
        return -2.5 * np.log10(j_comb)


def _get_logg_MS(teff_params):
    # # compute logg along main sequence for observed Galah stars
    # k_teff = -4.3e-4  # computed from two points on the Galah main sequence
    # n_logg = logg - k_teff * teff
    # y_logg = k_teff*teff_params + n_logg

    # quadratic model for the MS model, same for all objects
    if USE_IDR3:
        # iDR3 SME Results on complete dataset
        # y_logg = 3.616042 + 0.0003813298 * teff_params - 7.922988e-9 * teff_params ** 2 - 6.091324e-12 * teff_params ** 3
        y_logg = 2.575989 + 0.0009476907 * teff_params - 1.100047e-7 * teff_params ** 2
    else:
        # DR2 Cannon
        y_logg = -4.062806 + 0.003456557 * teff_params - 3.470085e-7 * teff_params ** 2
    return y_logg


def lnprob_mag_fit(params, feh, photo_obj, photo_obj_std, teff_range):
    if eval_params(params, teff_range):
        # compute logg along main sequence for observed Galah stars
        y_logg = _get_logg_MS(params)

        # get combined mag and compare it
        phot_comb = _comb_mag_photometry_precomputed(params, y_logg, feh, return_flux=False)
        # phot_comb = _comb_mag_photometry(params, y_logg, feh, return_flux=False)
        if len(phot_comb) == 0:
            return -np.inf

        # determine chi2
        # V1 - compute difference directly on magnitude values
        phot_diff = (photo_obj - phot_comb)**2
        lnprob_val = -10 * (np.nansum(phot_diff / photo_obj_std ** 2 + np.log(2. * np.pi * photo_obj_std ** 2)))
        # V2 - compute difference on flux
        # phot_diff = (2.5**(np.abs(photo_obj - phot_comb))) ** 2
        # lnprob_val = -100.*(np.nansum(phot_diff/(2.5**photo_obj_std)**2 + np.log(2.*np.pi*(2.5**photo_obj_std)**2)))

        # print lnprob_val, params, y_logg, feh
        # print phot_comb
        if np.isfinite(lnprob_val):
            return lnprob_val
        else:
            return -np.inf
    else:
        return -np.inf


def plot_corner_values_only(flatchain, write_out=True, path='figure.png'):
    plt.rcParams['font.size'] = 15
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(7, 5.5))
    ax[0, 0].plot([4000, 7000], [4000, 7000], c='black')
    ax[0, 0].scatter(flatchain[:, 0], flatchain[:, 1], c='black', lw=0, s=2.5, alpha=0.6)
    ax[0, 0].set(xlabel=u'T$_{eff1}$ [K]', ylabel=u'T$_{eff2}$ [K]', ylim=(5100, 6300), xlim=(5380, 6570),
                 xticks=[5400, 5600, 5800, 6000, 6200, 6400], xticklabels=[],
                 yticks=[5200, 5400, 5600, 5800, 6000, 6200], yticklabels=['5200', '', '5600', '', '6000', ''])
    ax[0, 0].grid(ls='--', alpha=0.2, color='black')

    ax[1, 0].plot([4000, 7000], [4000, 7000], c='black')
    ax[1, 0].scatter(flatchain[:, 0], flatchain[:, 2], c='black', lw=0, s=2.5, alpha=0.6)
    ax[1, 0].set(xlabel=u'T$_{eff1}$ [K]', ylabel=u'T$_{eff3}$ [K]', ylim=(4650, 5910), xlim=(5380, 6570),
                 xticks=[5400, 5600, 5800, 6000, 6200, 6400], xticklabels=['5400', '', '5800', '', '6200', ''],
                 yticks=[4800, 5000, 5200, 5400, 5600, 5800], yticklabels=['4800', '', '5200', '', '5600', ''])
    ax[1, 0].grid(ls='--', alpha=0.2, color='black')

    ax[1, 1].plot([4000, 7000], [4000, 7000], c='black')
    ax[1, 1].scatter(flatchain[:, 1], flatchain[:, 2], c='black', lw=0, s=2.5, alpha=0.6)
    ax[1, 1].set(xlabel=u'T$_{eff2}$ [K]', ylabel='', xlim=(5100, 6300), ylim=(4650, 5910),
                 xticks=[5200, 5400, 5600, 5800, 6000, 6200], xticklabels=['5200', '', '5600', '', '6000', ''],
                 yticks=[4800, 5000, 5200, 5400, 5600, 5800], yticklabels=[])
    ax[1, 1].grid(ls='--', alpha=0.2, color='black')

    # remove unused plot axis
    ax[0, 1].set_visible(False)

    if write_out:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(path, dpi=250)
    plt.close(fig)
    plt.rcParams['font.size'] = 12


def plot_corner_with_lnprob(flatchain, flatlnprob, path='', low_perc=90., write_out=True):
    vmin_lnp = np.nanpercentile(flatlnprob, low_perc)
    vmax_lnp = np.nanpercentile(flatlnprob, 100)+10
    flatlnprob_best = flatlnprob[flatlnprob >= vmin_lnp]
    flatchain_best = flatchain[np.where(flatlnprob >= vmin_lnp)[0], :]
    median_val = np.median(flatchain_best, axis=0)
    if flatchain.shape[1] == 1:
        fig, ax = plt.subplots(1, 1)
        x_range = np.percentile(flatchain[:, 0], [0.5, 99.5])
        ax.hist(flatchain[:, 0], range=x_range, bins=100, alpha=1.)
        ax.hist(flatchain_best[:, 0], range=x_range, bins=100, alpha=1.)
        ax.axvline(median_val[0], ls='--', color='red')
        ax.set(xlabel='Teff 1', ylabel='Distribution of values in flatchain')
    elif flatchain.shape[1] == 2:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(median_val[0], median_val[1], lw=0, s=75, c='red', marker='X')
        ax.scatter(flatchain[:, 0], flatchain[:, 1], c='black', lw=0, s=0.5, alpha=0.2)
        color_ax = ax.scatter(flatchain_best[:, 0], flatchain_best[:, 1], c=flatlnprob_best, lw=0, s=1, vmin=vmin_lnp, vmax=vmax_lnp)
        plt.colorbar(color_ax, ax=ax)
        ax.set(xlabel='Teff 1', ylabel='Teff 2')
    else:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].scatter(median_val[0], median_val[1], s=75, c='red', marker='X', lw=0)
        ax[0, 0].scatter(flatchain[:, 0], flatchain[:, 1], c='black', lw=0, s=0.5, alpha=0.2)
        color_ax = ax[0, 0].scatter(flatchain_best[:, 0], flatchain_best[:, 1], c=flatlnprob_best, lw=0, s=1, vmin=vmin_lnp, vmax=vmax_lnp)
        ax[0, 0].set(xlabel='Teff 1', ylabel='Teff 2')

        ax[1, 0].scatter(median_val[0], median_val[2], s=75, c='red', marker='X', lw=0)
        ax[1, 0].scatter(flatchain[:, 0], flatchain[:, 2], c='black', lw=0, s=0.5, alpha=0.2)
        ax[1, 0].scatter(flatchain_best[:, 0], flatchain_best[:, 2], c=flatlnprob_best, lw=0, s=1, vmin=vmin_lnp, vmax=vmax_lnp)
        ax[1, 0].set(xlabel='Teff 1', ylabel='Teff 3')

        ax[1, 1].scatter(median_val[1], median_val[2], s=75, c='red', marker='X', lw=0)
        ax[1, 1].scatter(flatchain[:, 1], flatchain[:, 2], c='black', lw=0, s=0.5, alpha=0.2)
        ax[1, 1].scatter(flatchain_best[:, 1], flatchain_best[:, 2], c=flatlnprob_best, lw=0, s=1, vmin=vmin_lnp, vmax=vmax_lnp)
        ax[1, 1].set(xlabel='Teff 2', ylabel='Teff 3')
        plt.colorbar(color_ax, ax=ax[0, 1])
    plt.tight_layout()
    if write_out:
        plt.savefig(path, dpi=400)
    plt.close(fig)


def plt_magnitudes(mag_1, mag_2, label_1, label_2, x_labels, path, mag_std_1=None, mag_std_2=None, write_out=True):
    x_p_pos = np.arange(len(x_labels))
    plt.errorbar(x_p_pos, mag_1, ms=6, label=label_1, yerr=mag_std_1, capsize=0, ls='None', fmt='.', mew=0, elinewidth=1)
    plt.errorbar(x_p_pos, mag_2, ms=6, label=label_2, yerr=mag_std_2, capsize=0, ls='None', fmt='.', mew=0, elinewidth=1)
    plt.xticks(x_p_pos, x_labels, rotation=90)
    plt.title('Median diff: {:.2f}       Chi2: {:.2f}'.format(np.nanmedian(mag_2 - mag_1), np.nansum((mag_1 - mag_2)**2/mag_std_1**2)))
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    if write_out:
        plt.savefig(path, dpi=200)
    plt.close()


def get_cannon(teff, logg, feh, vsini=None):
    sint = thetas[:, 0] * 0.0
    if vsini is None:
        labs = (np.array([teff, logg, feh]) - fid) / sca
    else:
        labs = (np.array([teff, logg, feh, vsini]) - fid) / sca
    vec = vectorizer(labs)
    for i, j in enumerate(vec):
        sint += thetas[:, i] * j
    return sint


def synthetic_spectra_combine(teff_vals, logg_vals, feh, mag_values):
    # define wavelength ranges of HERMES arms
    min_wvl = np.array([4705, 5640, 6470, 7680])
    max_wvl = np.array([4915, 5885, 6750, 7900])
    feh_use = feh[0]

    # determine vsini values for objects in the parameter vicinity of the requested
    d_teff = 80.
    d_logg = 0.05
    d_feh = 0.1
    idx_use = galah_gaia_data['flag_cannon'] == 0
    if idx_ok is not None:
        idx_use = np.logical_and(idx_use, idx_ok)
    vsini_vals = list([])
    for i_t in range(len(teff_vals)):
        # compute and add vsini
        idx_use_vsini = np.logical_and(idx_use, np.abs(galah_gaia_data['Teff_cannon'] - teff_vals[i_t]) < d_teff)
        idx_use_vsini = np.logical_and(idx_use_vsini, np.abs(galah_gaia_data['Logg_cannon'] - logg_vals[i_t]) < d_logg)
        idx_use_vsini = np.logical_and(idx_use_vsini, np.abs(galah_gaia_data['Fe_H_cannon'] - feh_use) < d_feh)
        if np.sum(idx_use_vsini) < 10:
            # use a bit wider parameter space to determine median vsini of the stars
            d_teff = 100.
            d_logg = 0.1
            idx_use_vsini = np.logical_and(idx_use, np.abs(galah_gaia_data['Teff_cannon'] - teff_vals[i_t]) < d_teff)
            idx_use_vsini = np.logical_and(idx_use_vsini, np.abs(galah_gaia_data['Logg_cannon'] - logg_vals[i_t]) < d_logg)
            idx_use_vsini = np.logical_and(idx_use_vsini, np.abs(galah_gaia_data['Fe_H_cannon'] - feh_use) < d_feh)
        vsini_vals.append(np.nanmedian(galah_gaia_data['Vsini_cannon'][idx_use_vsini]))

    flx_complete = list([])
    for ib in range(len(min_wvl)):
        idx_wvl_mask = np.logical_and(cannon_model.dispersion >= min_wvl[ib], cannon_model.dispersion <= max_wvl[ib])

        # flx_1 = get_cannon(teff_vals[0], logg_vals[0], feh_use)[idx_wvl_mask]
        flx_1 = get_cannon(teff_vals[0], logg_vals[0], feh_use, vsini=vsini_vals[0])[idx_wvl_mask]
        if len(teff_vals) == 1:
            flx_model = flx_1
        elif len(teff_vals) == 2:
            j_1 = 10 ** (-0.4 * mag_values[1][ib]) / 10 ** (-0.4 * mag_values[0][ib])
            # flx_2 = get_cannon(teff_vals[1], logg_vals[1], feh_use)[idx_wvl_mask]
            flx_2 = get_cannon(teff_vals[1], logg_vals[1], feh_use, vsini=vsini_vals[1])[idx_wvl_mask]
            flx_model = 1. / (1. + j_1) * flx_1 + 1. / (1. + (1. / j_1)) * flx_2
        elif len(teff_vals) == 3:
            j_1 = 10 ** (-0.4 * mag_values[1][ib]) / 10 ** (-0.4 * mag_values[0][ib])
            j_2 = 10 ** (-0.4 * mag_values[2][ib]) / 10 ** (-0.4 * mag_values[0][ib])
            # flx_2 = get_cannon(teff_vals[1], logg_vals[1], feh_use)[idx_wvl_mask]
            # flx_3 = get_cannon(teff_vals[2], logg_vals[2], feh_use)[idx_wvl_mask]
            flx_2 = get_cannon(teff_vals[1], logg_vals[1], feh_use, vsini=vsini_vals[1])[idx_wvl_mask]
            flx_3 = get_cannon(teff_vals[1], logg_vals[1], feh_use, vsini=vsini_vals[2])[idx_wvl_mask]
            flx_model = 1. / (1. + j_1 + j_2) * flx_1 + j_1 / (1. + j_1 + j_2) * flx_2 + j_2 / (1. + j_1 + j_2) * flx_3

        flx_complete.append(flx_model)

    return np.hstack(flx_complete)


def lnprob_flx_fit(feh, teff_model, logg_model, flx, flx_s, wvl_mask):
    if -0.5 < feh < 0.4:
        try:
            mag_values = _list_mag_photometry(teff_model, logg_model, feh, p_cols_galah)
            flx_star_model = synthetic_spectra_combine(teff_model, logg_model, feh, mag_values)[wvl_mask]
        except:
            return -np.inf
        lnprob_flux = -0.5 * (np.nansum((flx - flx_star_model) ** 2 / flx_s**2 + np.log(2*np.pi*flx_s**2)))
        if np.isfinite(lnprob_flux):
            return lnprob_flux
        else:
            return -np.inf
    else:
        return -np.inf


def teff_final_from_best_lnprob(flatlnprob, flatchain, percentile=80.):
    min_lnprob = np.nanpercentile(flatlnprob, percentile)
    idx_use_chainvla = np.where(flatlnprob >= min_lnprob)[0]
    if len(idx_use_chainvla) <= 0:
        idx_use_chainvla = np.where(flatlnprob >= np.nanpercentile(flatlnprob, percentile - 10.))[0]
    print '   - points used to eval teff:', len(idx_use_chainvla)
    return np.median(flatchain[idx_use_chainvla, :], axis=0)


def plt_feh_distribution(data, path, orig_feh=None, orig_feh_flag=None, write_out=True):
    plt.hist(data, bins=100, range=(-0.5, 0.5))
    median_feh = np.median(data)
    plt.axvline(median_feh, color='black', ls='--')
    if orig_feh is not None:
        plt.axvline(orig_feh, color='red', ls='--')
        title_add = ', change: {:.2f}'.format(median_feh - orig_feh)
    else:
        title_add = ''
    if orig_feh_flag is not None:
        if orig_feh_flag > 0:
            title_add += ', Cannon flag: {:.0f}'.format(orig_feh_flag)
    plt.title('[Fe/H] distribution for teff combination'+title_add)
    if write_out:
        plt.savefig(path, dpi=200)
    plt.close()


def fit_photometry_to_object(data_obj, flx, flx_s, wvl,
                             fit_single=False, fit_double=True, fit_tripple=True,
                             nwalkers=20, n_steps_1=50, n_steps_2=200, n_steps_feh=40, n_threds=10,
                             suffix='', save_pkl=True, write_out=True,
                             complete_wvl_range=False, fe_wvl_range_only=False):

    print ' Threads to be used:', n_threds
    if not write_out:
        print ' Omitting all outputs'
        save_pkl = False

    str_s_id = str(data_obj['sobject_id'][0])
    # obj_photo = (data_obj[p_cols].to_pandas().values - 2.5*np.log10(((1e3/data_obj['parallax'][0])/10.)**2))[0]
    obj_photo = (data_obj[p_cols].to_pandas().values - 2.5*np.log10(((data_obj['r_est'][0])/10.)**2))[0]
    # get photometry std and check valid std values
    obj_photo_std = np.full(len(p_cols_sigma), 0.)
    for i_s_c in range(len(p_cols_sigma)):
        if p_cols_sigma[i_s_c] in data_obj.colnames:
            obj_photo_std[i_s_c] = data_obj[p_cols_sigma[i_s_c]][0]
    obj_photo_std[obj_photo_std <= 0.] = np.median(obj_photo_std[obj_photo_std > 0.])
    n_photo_finite = np.sum(np.isfinite(obj_photo))
    print ' Ok phot:', n_photo_finite, 'out of', len(obj_photo)
    if n_photo_finite < 3:
        print '  TERMINATED: not enough photometric data to compute anything'
        return np.full(24, np.nan)
    # med_photo_cannonparams = median_photometry_precomputed_intepol(data_obj['Teff_cannon'], data_obj['Logg_cannon'], data_obj['Fe_H_cannon_orig'], p_cols).to_pandas().values[0]
    med_photo_cannonparams = median_photometry(data_obj['Teff_cannon'], data_obj['Logg_cannon'], data_obj['Fe_H_cannon_orig'], p_cols, d_teff=80, d_logg=0.05, d_feh=0.1, idx_init=idx_ok, min_data=10)

    # output plot of magnitudes
    if write_out:
        plt_magnitudes(obj_photo, med_photo_cannonparams, 'Observed', 'Median photo Cannon', p_cols,  str_s_id + '_aphot' + suffix + '_1.png',
                       mag_std_1=obj_photo_std, write_out=write_out)

    # prepare original spectra
    # define used subset of the wvl data
    if complete_wvl_range:
        idx_cannon_wvl_mask = cannon_model.dispersion > 0
    else:
        if fe_wvl_range_only:
            idx_cannon_wvl_mask = get_linelist_mask(cannon_model.dispersion, d_wvl=0., element='Fe')
        else:
            idx_cannon_wvl_mask = get_linelist_mask(cannon_model.dispersion, d_wvl=0.)
    idx_cannon_wvl_feh = get_linelist_mask(cannon_model.dispersion, d_wvl=0., element='Fe')

    # resample read spectra to the same wvl pixels as cannon mask
    wvl_new = cannon_model.dispersion[idx_cannon_wvl_mask]
    flx_new = spectra_resample(flx, wvl, wvl_new, k=1)
    flx_s_new = spectra_resample(flx_s, wvl, wvl_new, k=1)

    # resample read spectra to the same wvl pixels as feh mask
    wvl_new_feh = cannon_model.dispersion[idx_cannon_wvl_feh]
    flx_new_feh = spectra_resample(flx, wvl, wvl_new_feh, k=1)
    flx_s_new_feh = spectra_resample(flx_s, wvl, wvl_new_feh, k=1)

    def _p0_generate(teff_init_range, mean_val, n_walkers, n_instances=3):
        p0 = list([])
        for i_w in range(n_walkers):
            t_vals = mean_val - teff_init_range / 2. + np.random.rand(n_instances) * teff_init_range
            p0_new = np.sort(t_vals)[::-1]
            p0.append(p0_new)
        return p0

    def _p0_perturbe(p0_vals, perc=2.):
        p0_shape = p0_vals.shape
        p0_pertrubed = p0_vals + p0_vals*(np.random.uniform(-perc, perc, size=p0_shape)/100.)
        return p0_pertrubed

    def run_teff_mcmc(p0, feh_use, n_s, n_t):
        t_1 = time()
        sampler = emcee.EnsembleSampler(len(p0), len(p0[0]), lnprob_mag_fit, threads=n_t,
                                        args=(feh_use, obj_photo, obj_photo_std, [4700, 6400]))
        try:
            p0, lnp, _ = sampler.run_mcmc(p0, n_s)
            sampler.pool.close()
            sampler.pool = None
            print '   - took {:.2f} min'.format((time() - t_1) / 60.)
            return sampler
        except ValueError:
            sampler.pool.close()
            sampler.pool = None
            sampler.reset()
            return None

    def run_flux_mcmc(p0, teff_list, logg_list, n_s, n_t):
        t_1 = time()
        sampler = emcee.EnsembleSampler(len(p0), 1, lnprob_flx_fit, threads=n_t,
                                        args=(teff_list, logg_list, flx_new_feh, flx_s_new_feh, idx_cannon_wvl_feh))
        try:
            p0, lnp, _ = sampler.run_mcmc(p0, n_s)
            sampler.pool.close()
            sampler.pool = None
            print '   - took {:.2f} min'.format((time() - t_1) / 60.)
            return sampler
        except ValueError:
            sampler.pool.close()
            sampler.pool = None
            sampler.reset()
            return None

    def run_complete_fit_procedure(n_stars, write_out=True):
        data_obj_stars = deepcopy(data_obj)
        # print ' Initial [Fe/H]:'
        # print '', data_obj_stars['Fe_H_cannon', 'Fe_H_cannon_orig']
        # init samples for 2 star fit
        out_file_pkl = str_s_id + suffix + '_s'+str(n_stars)+'_0.pkl'

        if not save_pkl:
            run_mcmc = True
        else:
            run_mcmc = not path.isfile(out_file_pkl)
        if run_mcmc:
            if n_stars == 1:
                n_steps_init = 150
                w_m_b = 5
                p0_1 = _p0_generate(1000, np.array([data_obj_stars['Teff_cannon'][0]]), w_m_b*nwalkers, n_instances=n_stars)
            elif n_stars == 2:
                n_steps_init = 175
                w_m_b = 7
                p0_1 = _p0_generate(1100, np.array([data_obj_stars['Teff_cannon'][0] + 300,
                                                    data_obj_stars['Teff_cannon'][0] - 300.]), w_m_b*nwalkers, n_instances=n_stars)
            elif n_stars == 3:
                n_steps_init = 200
                w_m_b = 8
                p0_1 = _p0_generate(1100, np.array([data_obj_stars['Teff_cannon'][0] + 300,
                                                    data_obj_stars['Teff_cannon'][0],
                                                    data_obj_stars['Teff_cannon'][0] - 400.]), w_m_b*nwalkers, n_instances=n_stars)

            # plot initial walker values
            plot_corner_values_only(np.array(p0_1), write_out=True,
                                    path=str_s_id + suffix + '_init' + '_' + str(n_stars) + '.png')

            print '  Initial MCMC burn ('+str(n_stars)+' stars) - {:.0f} steps'.format(n_steps_init)
            sampler = run_teff_mcmc(p0_1, data_obj_stars['Fe_H_cannon'][0], n_steps_init, n_threds)
            if save_pkl and write_out:
                joblib.dump(sampler, out_file_pkl)
        else:
            # save chain and lnprob for later use
            sampler = joblib.load(out_file_pkl)
        if write_out:
            # other plots
            teff_s0_stars = teff_final_from_best_lnprob(sampler.flatlnprobability, sampler.flatchain, percentile=85.)
            plot_lnprob(sampler.flatchain, teff_s0_stars, str_s_id + suffix + '_corner' + '_'+str(n_stars)+'star_00.png', write_out=write_out)
            plot_corner_with_lnprob(sampler.flatchain, sampler.flatlnprobability,
                                    path=str_s_id + suffix + '_corner' + '_' + str(n_stars) + 'star_0.png', write_out=write_out)
            plot_walkers(sampler.lnprobability, str_s_id + suffix + '_lnprob' + '_'+str(n_stars)+'star_0.png',
                         write_out=write_out)

        # select and pertrubate the best from initial burn
        out_file_pkl = str_s_id + suffix + '_s' + str(n_stars) + '_1.pkl'
        if not save_pkl:
            run_mcmc = True
        else:
            run_mcmc = not path.isfile(out_file_pkl)
        if run_mcmc:
            # select walkers from the last run
            ln = sampler.lnprobability[:, -1]
            # use only the best ones
            idx_ln_best = np.argsort(ln)[::-1][:nwalkers]
            p0_1 = sampler.chain[idx_ln_best, -1, :]
            print '  First MCMC run (' + str(n_stars) + ' stars) - {:.0f} steps'.format(n_steps_1)
            # run with original walkers
            # sampler = run_teff_mcmc(p0_1, data_obj_stars['Fe_H_cannon'][0], n_steps_1, n_threds)
            # run with pertrubed walkers
            print '    Running with pertrubed walkers.'
            sampler = run_teff_mcmc(_p0_perturbe(p0_1), data_obj_stars['Fe_H_cannon'][0], n_steps_1, n_threds)
            if save_pkl and write_out:
                joblib.dump(sampler, out_file_pkl)
        else:
            # save chain and lnprob for later use
            sampler = joblib.load(out_file_pkl)

        # evaluate results from the initial burn, compute new priors accordingly based on the best lnprobs in the last few steps
        teff_s1_stars = teff_final_from_best_lnprob(sampler.flatlnprobability, sampler.flatchain, percentile=90.)
        print '  Intermediate teff:', teff_s1_stars
        # plot lnprobs and walkers
        if write_out:
            plot_lnprob(sampler.flatchain, teff_s1_stars, str_s_id + suffix + '_corner' + '_'+str(n_stars)+'star_2.png', write_out=write_out)
            plot_walkers(sampler.lnprobability, str_s_id + suffix + '_lnprob' + '_'+str(n_stars)+'star_2.png')
            plot_corner_with_lnprob(sampler.flatchain, sampler.flatlnprobability,
                                    path=str_s_id + suffix + '_corner' + '_'+str(n_stars)+'star_1.png', write_out=write_out)
        y_logg_stars = _get_logg_MS(teff_s1_stars)
        sampler.reset()

        step1_mag_stars = _comb_mag_photometry_precomputed(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'])
        # step1_mag_stars = _comb_mag_photometry(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'])
        if write_out:
            plt_magnitudes(obj_photo, step1_mag_stars, 'Observed', 'Fitted', p_cols,
                           str_s_id + suffix + '_aphot' + '_'+str(n_stars)+'star_1-nofeh.png',
                           mag_std_1=obj_photo_std, write_out=write_out)

        # determine best matching metalicity for the selected teff combination
        p0_feh = _p0_generate(0.4, np.array([data_obj_stars['Fe_H_cannon'][0]]), nwalkers, n_instances=1)
        print '  Fe/H MCMC run ('+str(n_stars)+' stars) - {:.0f} steps'.format(n_steps_feh)
        sampler = run_flux_mcmc(p0_feh, teff_s1_stars, y_logg_stars, n_steps_feh, n_threds)
        if sampler is None:
            feh_stars = np.nan
            model_ok_stars = False
        else:
            feh_stars = np.nanmedian(sampler.chain)
            print '   '+str(n_stars)+' star Feh:', feh_stars
            if write_out:
                plt_feh_distribution(sampler.flatchain, str_s_id + suffix + '_feh' + '_'+str(n_stars)+'star.png',
                                     orig_feh=data_obj_stars['Fe_H_cannon'][0], orig_feh_flag=data_obj['flag_cannon'][0], write_out=write_out)
            sampler.reset()

        # set feh of the objects
        data_obj_stars['Fe_H_cannon'] = feh_stars

        if np.isfinite(feh_stars):
            # final fit for for star fit
            p0_1 = _p0_generate(100, np.array(teff_s1_stars), nwalkers, n_instances=n_stars)
            print '  Second and final MCMC run ('+str(n_stars)+' stars) - {:.0f} steps'.format(n_steps_2)
            sampler = run_teff_mcmc(p0_1, data_obj_stars['Fe_H_cannon'][0], n_steps_2, n_threds)
            # evaluate results from the last burn
            teff_s1_stars = teff_final_from_best_lnprob(sampler.flatlnprobability, sampler.flatchain, percentile=70.)
            # plot lnprobs and walkers
            if write_out:
                plot_lnprob(sampler.flatchain, teff_s1_stars, str_s_id + suffix + '_corner' + '_'+str(n_stars)+'star_final.png', write_out=write_out)
                plot_walkers(sampler.lnprobability, str_s_id + suffix + '_lnprob' + '_'+str(n_stars)+'star_final.png')
            y_logg_stars = _get_logg_MS(teff_s1_stars)
            sampler.reset()

            # check if a star model is even possible within the Cannon limitation
            model_ok_stars = True

            try:
                final_mag_stars = _comb_mag_photometry_precomputed(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'])
                # final_mag_stars = _comb_mag_photometry(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'])
                if len(final_mag_stars) == 0:
                    model_ok_stars = False
                if write_out:
                    plt_magnitudes(obj_photo, final_mag_stars, 'Observed', 'Fitted', p_cols,
                                   str_s_id + suffix + '_aphot' + '_'+str(n_stars)+'star_2.png',
                                   mag_std_1=obj_photo_std, write_out=write_out)
            except:
                final_mag_stars = np.full_like(step1_mag_stars, np.nan)  # np.full(len(p_cols), np.nan)
                model_ok_stars = False
                plt.close()

        flx_onestar = synthetic_spectra_combine(data_obj_stars['Teff_cannon'], data_obj_stars['Logg_cannon'], data_obj_stars['Fe_H_cannon_orig'], None)[idx_cannon_wvl_mask]
        sim_f_onestar = np.nansum((flx_new - flx_onestar) ** 2 / flx_s_new ** 2)

        sim_p0_chi = np.nansum((obj_photo - med_photo_cannonparams) ** 2 / obj_photo_std ** 2)
        sim_p0_exc = np.nanmedian(med_photo_cannonparams - obj_photo)

        # compute and return similarity values
        if model_ok_stars:
            mag_values_stars = _list_mag_photometry(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'], p_cols_galah)
            flx_stars = synthetic_spectra_combine(teff_s1_stars, y_logg_stars, data_obj_stars['Fe_H_cannon'], mag_values_stars)[idx_cannon_wvl_mask]

            # similarity computation - flux
            sim_f = np.nansum((flx_new - flx_stars) ** 2 / flx_s_new ** 2)

            # similarity computation - photometry
            sim_p2 = np.nansum((obj_photo - final_mag_stars) ** 2 / obj_photo_std ** 2)

        else:
            flx_stars = np.full_like(flx_new, np.nan)
            sim_f = np.nan
            sim_p2 = np.nan

        # multiple similarities, just for check
        idx_1 = get_linelist_mask(wvl_new, d_wvl=0.)
        idx_2 = get_linelist_mask(wvl_new, d_wvl=0., element='Fe')
        print ' Sim fe, abs, all, onestar:  ',  np.nansum((flx_new[idx_2] - flx_stars[idx_2])**2/flx_s_new[idx_2]**2), np.nansum((flx_new[idx_1] - flx_stars[idx_1])**2/flx_s_new[idx_1]**2), sim_f, sim_f_onestar

        return flx_stars, np.hstack((teff_s1_stars, feh_stars, sim_p0_exc, sim_p0_chi, sim_p2, sim_f_onestar, sim_f))

    # --------------------------------
    # ------- Run fits for selected number of stars in the configuration
    # --------------------------------
    # single star fit
    if fit_single:
        s1_flx, s1_fit_final = run_complete_fit_procedure(1, write_out=write_out)
    else:
        s1_fit_final = np.full(7, np.nan)
        s1_flx = np.full_like(flx_new, np.nan)
    # double star fit
    if fit_double:
        s2_flx, s2_fit_final = run_complete_fit_procedure(2, write_out=write_out)
    else:
        s2_fit_final = np.full(8, np.nan)
        s2_flx = np.full_like(flx_new, np.nan)
    # triple star fit
    if fit_tripple:
        s3_flx, s3_fit_final = run_complete_fit_procedure(3, write_out=write_out)
    else:
        s3_fit_final = np.full(9, np.nan)
        s3_flx = np.full_like(flx_new, np.nan)

    # --------------------------------
    # ------- Outputs, table, plots --
    # --------------------------------
    if write_out:
        flx_onestar = synthetic_spectra_combine(data_obj['Teff_cannon'], data_obj['Logg_cannon'], data_obj['Fe_H_cannon'], None)[idx_cannon_wvl_mask]
        if complete_wvl_range:
            plt.figure(figsize=(35, 5))
        else:
            plt.figure(figsize=(17, 5))
        plt.plot(flx_new, label='Observed', lw=0.5, c='black')
        plt.plot(s1_flx, label='1 star model', lw=0.5, c='C0')
        plt.plot(s2_flx, label='2 star model', lw=0.5, c='C1')
        plt.plot(s3_flx, label='3 star model', lw=0.5, c='C2')
        plt.plot(flx_onestar, label='Params star model', lw=0.5, c='C3')
        plt.title('Cannon paramas: {:.0f}  {:.2f}  {:.2f},    similarity    1 star: {:.2f}    2 stars: {:.2f}    3 stars: {:.2f}    params: {:.2f}'.format(data_obj['Teff_cannon'][0], data_obj['Logg_cannon'][0], data_obj['Fe_H_cannon_orig'][0], s1_fit_final[-1], s2_fit_final[-1], s3_fit_final[-1], s1_fit_final[-2]))
        plt.tight_layout()
        plt.xlim(0, len(flx_new))
        plt.ylim(0.3, 1.02)
        plt.legend()
        plt.savefig(str_s_id + suffix + '_spectra' + '_all.png', dpi=300)
        plt.close()

    _, n_binary_photo2, n_binary_spectra = determine_number_of_star(s1_fit_final, s2_fit_final, s3_fit_final)

    return np.hstack((s1_fit_final, s2_fit_final, s3_fit_final,
                      n_binary_photo2, n_binary_spectra))


def determine_number_of_star(s1_fit_final, s2_fit_final, s3_fit_final):
    # --------------------------------
    # ------- Determine the best combination for the photometry and spectroscopy
    # --------------------------------
    # determine number of stars
    if np.sum(np.isfinite([s1_fit_final[-3], s2_fit_final[-3], s3_fit_final[-3]])) == 0:
        n_binary_photo2 = np.nan
    else:
        n_binary_photo2 = np.nanargmin([s1_fit_final[-3], s2_fit_final[-3], s3_fit_final[-3]]) + 1

    if np.sum(np.isfinite([s1_fit_final[-1], s2_fit_final[-1], s3_fit_final[-1]])) == 0:
        n_binary_spectra = np.nan
    else:
        n_binary_spectra = np.nanargmin([s1_fit_final[-1], s2_fit_final[-1], s3_fit_final[-1]]) + 1

    return np.nan, n_binary_photo2, n_binary_spectra


def fit_photometry_results_recompute(data_obj, flx, flx_s, wvl, fit_results,
                                     suffix='', write_out=True,
                                     fit_single=True, fit_double=True, fit_tripple=True):

    str_s_id = str(data_obj['sobject_id'][0])
    obj_photo = (data_obj[p_cols].to_pandas().values - 2.5*np.log10(((data_obj['r_est'][0])/10.)**2))[0]
    # get photometry std and check valid std values
    obj_photo_std = np.full(len(p_cols_sigma), 0.)
    for i_s_c in range(len(p_cols_sigma)):
        if p_cols_sigma[i_s_c] in data_obj.colnames:
            obj_photo_std[i_s_c] = data_obj[p_cols_sigma[i_s_c]][0]
    obj_photo_std[obj_photo_std <= 0.] = np.median(obj_photo_std[obj_photo_std > 0.])
    n_photo_finite = np.sum(np.isfinite(obj_photo))
    print ' Ok phot:', n_photo_finite, 'out of', len(obj_photo)
    if n_photo_finite < 3:
        print '  TERMINATED: not enough photometric data to compute anything'
        return np.full(24, np.nan)
    # med_photo_cannonparams = median_photometry_precomputed_intepol(data_obj['Teff_cannon'], data_obj['Logg_cannon'], data_obj['Fe_H_cannon'], p_cols).to_pandas().values[0]
    med_photo_cannonparams = median_photometry(data_obj['Teff_cannon'], data_obj['Logg_cannon'], data_obj['Fe_H_cannon'], p_cols, d_teff=80, d_logg=0.05, d_feh=0.1, idx_init=idx_ok, min_data=10)
    # output plot of magnitudes
    if write_out:
        plt_magnitudes(obj_photo, med_photo_cannonparams, 'Observed', 'Median photo Cannon', p_cols,  str_s_id + '_aphot' + suffix + '_1.png',
                       mag_std_1=obj_photo_std, write_out=write_out)

    # prepare original spectra
    # define used subset of the wvl data
    idx_cannon_wvl_mask = cannon_model.dispersion > 0

    # resample read spectra to the same wvl pixels as cannon mask
    wvl_new = cannon_model.dispersion[idx_cannon_wvl_mask]
    flx_new = spectra_resample(flx, wvl, wvl_new, k=1)
    flx_s_new = spectra_resample(flx_s, wvl, wvl_new, k=1)

    def run_complete_results_recompute(teff_s1_stars, feh_val, write_out=True):
        print 'Recomputing fit for:', teff_s1_stars, feh_val
        n_stars = len(teff_s1_stars)
        y_logg_stars = _get_logg_MS(teff_s1_stars)
        # check if 2 star model is even possible within the Cannon limitation
        model_ok_stars = True
        try:
            final_mag_stars = _comb_mag_photometry_precomputed(teff_s1_stars, y_logg_stars, feh_val)
            if len(final_mag_stars) == 0:
                model_ok_stars = False
            if write_out:
                plt_magnitudes(obj_photo, final_mag_stars, 'Observed', 'Fitted', p_cols,
                               str_s_id + suffix + '_aphot' + '_'+str(n_stars)+'star_2.png',
                               mag_std_1=obj_photo_std, write_out=write_out)
        except:
            final_mag_stars = np.full_like(med_photo_cannonparams, np.nan)  # np.full(len(p_cols), np.nan)
            model_ok_stars = False
            plt.close()

        # compute and return similarity values
        if model_ok_stars:
            mag_values_stars = _list_mag_photometry(teff_s1_stars, y_logg_stars, feh_val, p_cols_galah)
            flx_stars = synthetic_spectra_combine(teff_s1_stars, y_logg_stars, feh_val, mag_values_stars)[idx_cannon_wvl_mask]

            # similarity computation - flux
            flx_dif_w = (flx_new - flx_stars) ** 2  # / flx_s_new ** 2
            flx_dif_w = flx_dif_w[np.isfinite(flx_dif_w)]
            sim_f = np.sqrt(np.nansum(np.sort(flx_dif_w)[:-10]))
            # similarity computation - photometry
            # sim_p0 = np.sqrt(np.nansum((obj_photo - med_photo_cannonparams) ** 2 / obj_photo_std ** 2))
            sim_p0 = np.nanmedian(med_photo_cannonparams - obj_photo)
            sim_p2 = np.sqrt(np.nansum((obj_photo - final_mag_stars) ** 2 / obj_photo_std ** 2))

        else:
            flx_stars = np.full_like(flx_new, np.nan)
            sim_f = np.nan
            # sim_p0 = np.sqrt(np.nansum((obj_photo - med_photo_cannonparams)**2 / obj_photo_std**2))
            sim_p0 = np.nanmedian(med_photo_cannonparams - obj_photo)
            sim_p2 = np.nan

        return flx_stars, np.hstack((teff_s1_stars, feh_val, sim_p0, sim_p2, sim_f))

    # single star fit
    if fit_single:
        teff_vals = np.array([fit_results['s1_teff1']])
        feh_val = fit_results['s1_feh']
        s1_flx, s1_fit_final = run_complete_results_recompute(teff_vals, [feh_val], write_out=write_out)
    else:
        s1_fit_final = np.full(5, np.nan)
        s1_flx = np.full_like(flx_new, np.nan)
    # double star fit
    if fit_double:
        teff_vals = np.array([fit_results['s2_teff1'], fit_results['s2_teff2']])
        feh_val = fit_results['s2_feh']
        s2_flx, s2_fit_final = run_complete_results_recompute(teff_vals, [feh_val], write_out=write_out)
    else:
        s2_fit_final = np.full(6, np.nan)
        s2_flx = np.full_like(flx_new, np.nan)
    # triple star fit
    if fit_tripple:
        teff_vals = np.array([fit_results['s3_teff1'], fit_results['s3_teff2'], fit_results['s3_teff3']])
        feh_val = fit_results['s3_feh']
        s3_flx, s3_fit_final = run_complete_results_recompute(teff_vals, [feh_val], write_out=write_out)
    else:
        s3_fit_final = np.full(7, np.nan)
        s3_flx = np.full_like(flx_new, np.nan)

    # --------------------------------
    # ------- Outputs, table, plots --
    # --------------------------------
    if write_out:
        fig, ax = plt.subplots(2, 1, figsize=(35, 9), sharex=True)
        ax[0].plot(flx_new, label='Observed', lw=0.5, c='black')
        # ax[1].plot(np.sqrt((flx_new - s1_flx) ** 2 / flx_s_new ** 2), c='C0', label='')
        # ax[1].plot(np.sqrt((flx_new - s2_flx) ** 2 / flx_s_new ** 2), c='C1', label='')
        # ax[1].plot(np.sqrt((flx_new - s3_flx) ** 2 / flx_s_new ** 2), c='C2', label='')
        ax[1].plot((flx_new - s1_flx) ** 2, c='C0', label='')
        ax[1].plot((flx_new - s2_flx) ** 2, c='C1', label='')
        ax[1].plot((flx_new - s3_flx) ** 2, c='C2', label='')
        ax[0].plot(s1_flx, label='1 star model', lw=0.5, c='C0')
        ax[0].plot(s2_flx, label='2 star model', lw=0.5, c='C1')
        ax[0].plot(s3_flx, label='3 star model', lw=0.5, c='C2')
        ax[0].set(title='Cannon paramas: {:.0f}  {:.2f}  {:.2f},    similarity    1 star: {:.2f}    2 stars: {:.2f}    3 stars: {:.2f}'.format(data_obj['Teff_cannon'][0], data_obj['Logg_cannon'][0], data_obj['Fe_H_cannon'][0], s1_fit_final[-1], s2_fit_final[-1], s3_fit_final[-1]))
        ax[0].set(xlim=(0, np.sum(np.isfinite(flx_new))), ylim=(0.3, 1.02))
        ax[0].legend()
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(str_s_id + suffix + '_spectra' + '_all.png', dpi=300)
        plt.close()

    _, n_binary_photo2, n_binary_spectra = determine_number_of_star(s1_fit_final, s2_fit_final, s3_fit_final)

    return np.hstack((s1_fit_final, s2_fit_final, s3_fit_final,
                      n_binary_photo2, n_binary_spectra))
