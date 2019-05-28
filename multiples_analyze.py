from sys import argv
from getopt import getopt
from os import system, chdir
from multiples_analyze_functions import *
from multiples_analyze_functions import _get_logg_MS, _list_mag_photometry

imp.load_source('gaia_twins_photometry_functions', '../Solar-spectral-siblings/gaia_twins_photometry_functions.py')
from gaia_twins_photometry_functions import fit_MS_lin_line

# read and parse input arguments if passed to this py procedure
output_dir = 'Mag_fit'
RECOMPUTE = False
RECOMPUTE_FINAL = False
RUN_SINGLE_STARS = False
feh_init = None
params_run = None
params_run_magthr = None

process_obj_begin = 0
process_obj_end = 50000
if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['obj_beg=', 'obj_end=', 'dir=', 'feh=', 'single=', 'params=', 'magthr='])
    # set parameters, depending on user inputs
    print opts
    for o, a in opts:
        if o == '--obj_beg':
            process_obj_begin = np.int64(a)
        if o == '--obj_end':
            process_obj_end = np.int64(a)
        if o == '--dir':
            output_dir = str(a)
        if o == '--feh':
            feh_init = np.float64(a)
        if o == '--single':
            sing_int = np.int32(a)
            if sing_int > 0:
                RUN_SINGLE_STARS = True
        if o == '--params':
            params_run = a
        if o == '--magthr':
            params_run_magthr = np.float64(a)

envelope_sel = '07'
if params_run is not None and params_run_magthr is not None:
    # read list of objects for selected parameter suffix
    list_objects_read = get_sobjects_from_final_selection(out_dir_root + 'Distances_Step1_p0_SNRsamples0_ext4_oklinesonly_G20180327_C181221_withH_refpar_' + params_run +'/final_selection_' + envelope_sel + '_envelope.txt')
else:
    # read list of objects for Solar-twins
    list_objects_read = get_sobjects_from_final_selection(out_dir_root + 'Distances_Step1_p0_SNRsamples0_ext0_oklinesonly_G20180327_C181221_withH/final_selection_' + envelope_sel+ '_envelope.txt')

# determine which objects are to be analysed
galah_gaia_sub = galah_gaia_data[np.in1d(galah_gaia_data['sobject_id'], list_objects_read)]['sobject_id', 'parallax', 'phot_g_mean_mag', 'r_est', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
# abs_mags = galah_gaia_sub['phot_g_mean_mag'] - 2.5 * np.log10(((1e3 / galah_gaia_sub['parallax']) / 10.) ** 2)
abs_mags = galah_gaia_sub['phot_g_mean_mag'] - 2.5 * np.log10(((galah_gaia_sub['r_est']) / 10.) ** 2)
bp_rp_mags = galah_gaia_sub['phot_bp_mean_mag'] - galah_gaia_sub['phot_rp_mean_mag']

# do a MS fit on selected objects
bin_bp_rb = 0.25
ms_lin_line = fit_MS_lin_line(abs_mags, bp_rp_mags, path=None, d_above=bin_bp_rb)

if RUN_SINGLE_STARS:
    # select stars bellow the binary line
    list_objects = galah_gaia_sub['sobject_id'][abs_mags > (ms_lin_line(bp_rp_mags) - bin_bp_rb)]
else:
    # select stars above the binary line
    list_objects = galah_gaia_sub['sobject_id'][abs_mags <= (ms_lin_line(bp_rp_mags) - bin_bp_rb)]

print 'Number of selected objects:', len(list_objects)

system('mkdir '+out_dir_root+output_dir)
chdir(out_dir_root+output_dir)

table_out_fits = 'fit_results.fits'
if path.isfile(table_out_fits):
    print 'Reading previous results'
    table_out = Table.read(table_out_fits)
else:  # sim_p0_exc, sim_p0_chi, sim_p2, sim_f_onestar, sim_f
    s1_cols = ['s1_teff1', 's1_feh', 'phot_excs', 'phot_chi2', 's1_sim_p', 'spec_chi2', 's1_sim_f']
    s2_cols = ['s2_teff1', 's2_teff2', 's2_feh', 'phot_excs2', 'phot_chi22', 's2_sim_p', 'spec_chi22', 's2_sim_f']
    s3_cols = ['s3_teff1', 's3_teff2', 's3_teff3', 's3_feh', 'phot_excs3', 'phot_chi23', 's3_sim_p', 'spec_chi23', 's3_sim_f']
    all_cols = np.hstack((s1_cols, s2_cols, s3_cols))
    all_dtype = np.full(len(all_cols), 'float64', dtype=np.dtype('S10'))
    table_out = Table(names=np.hstack(('sobject_id', 'parallax', all_cols, 'n_stars_p', 'n_stars_f')),
                      dtype=np.hstack(('int64', 'float64', all_dtype, 'int32', 'int32')))

for s_id in [150408004101169]:#list_objects[process_obj_begin: process_obj_end]:
    print '======================================================'
    print '======================================================'
    print '======================================================'
    obj_data = galah_gaia_data[galah_gaia_data['sobject_id'] == s_id]
    if len(obj_data) > 1:
        obj_data.remove_rows([1])
    parallax_orig = obj_data['parallax'][0]
    parallax_std = obj_data['parallax_error'][0]
    print 'Working on:', s_id, '(parallax: {:.3f}, parallax_error: {:.3f}, r_est: {:.1f})'.format(parallax_orig, parallax_std, obj_data['r_est'][0])

    print obj_data['flag_cannon', 'Teff_cannon', 'Logg_cannon', 'Fe_H_cannon']
    if obj_data['flag_cannon'] != 0:
        print ' WARNING - Possibly problematic Cannon object'

    idx_sids = table_out['sobject_id'] == s_id
    n_sids = np.sum(idx_sids)
    if n_sids > 0 and not (RECOMPUTE or RECOMPUTE_FINAL):
        print ' Already processed'
        continue

    sub_dir = str(s_id)
    system('mkdir ' + sub_dir)
    chdir(sub_dir)

    flx, flx_s, wvl = get_spectra_complete(s_id, get_bands=[1, 2, 3])
    # filter out strange uncertanties
    # max_flx_s = np.percentile(np.abs(flx_s), 95)
    max_flx_s = 5 * np.median(np.abs(flx_s))
    idx_strange = np.abs(flx_s) > max_flx_s
    n_strange = np.sum(idx_strange)
    if n_strange > 0:
        print 'Flx_s corrected:', n_strange
        flx_s[idx_strange] = max_flx_s
    idx_strange = flx > 1.1
    n_strange = np.sum(idx_strange)
    if n_strange > 0:
        print 'Flx corrected:', n_strange
        flx[idx_strange] = 1.1

    # set obj feh to the init value if set, else use cannon determined feh values
    obj_data['Fe_H_cannon_orig'] = obj_data['Fe_H_cannon']
    if feh_init is not None:
        obj_data['Fe_H_cannon'] = feh_init

    n_parallax = 1
    if n_parallax <= 1:
        # in the case of computation at only one distance of the source
        parallax_sim = [obj_data['parallax'][0]]
    else:
        # in the case of computation at multiple source distances
        if n_sids == 0 and not (RECOMPUTE or RECOMPUTE_FINAL):
            parallax_sim = np.sort(np.random.normal(parallax_orig, parallax_std, n_parallax))  # sort from farthest to nearest
        else:
            if n_sids == 0:
                parallax_sim = np.sort(np.random.normal(parallax_orig, parallax_std, n_parallax))
            else:
                print '  Reusing previously determined parallax distribution'
                parallax_sim = table_out['parallax'][idx_sids]

    # remove old results
    if n_sids > 0 and (RECOMPUTE or RECOMPUTE_FINAL):
        table_old_res = table_out[np.where(idx_sids)[0]]
        table_out.remove_rows(np.where(idx_sids)[0])
        print 'Removed rows from results:', np.sum(idx_sids)

    if RECOMPUTE_FINAL:
        for old_res in table_old_res:
            parallax_use = obj_data['parallax'][0]
            suffix_use = '_p{:.3f}'.format(parallax_use)
            # run process to recompute final fitted parameters
            s_time = time()
            fit_res_all = fit_photometry_results_recompute(obj_data, flx, flx_s, wvl, old_res,
                                                           suffix=suffix_use, write_out=True,
                                                           fit_single=True, fit_double=True, fit_tripple=True)

            print 'Recompute results time: {:.1f} s'.format((time() - s_time)), '  parallax used:', parallax_use
            # store results to array
            table_out.add_row(np.hstack((s_id, parallax_use, fit_res_all)))
    else:
        for parallax_use in parallax_sim:
            obj_data['parallax'] = parallax_use
            suffix_use = '_p{:.3f}'.format(parallax_use)
            # run the complete fitting procedure
            s_time = time()
            fit_res_all = fit_photometry_to_object(obj_data, flx, flx_s, wvl, suffix=suffix_use,  # input data, suffix based on parallax repeat number
                                                   fit_single=False, fit_double=False, fit_tripple=True,  # which stellar configuration to fit
                                                   complete_wvl_range=True, fe_wvl_range_only=False, write_out=True, save_pkl=False,
                                                   nwalkers=150, n_steps_1=200, n_steps_2=200, n_steps_feh=100, n_threds=240) # number of fitting steps in MCMC
                                                   # nwalkers=160, n_steps_1=10, n_steps_2=10, n_steps_feh=10, n_threds=10)

            print 'Fit time: {:.1f} min'.format((time()-s_time)/60.), '  parallax used:', parallax_use
            # store results to array
            table_out.add_row(np.hstack((s_id, parallax_use, fit_res_all)))

    # write out results to fits file
    chdir('..')
    table_out.write(table_out_fits, overwrite=True)
