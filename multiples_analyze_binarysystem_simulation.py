# import matplotlib.pyplot as plt
from os import system, chdir
from sys import argv
from getopt import getopt
from multiples_analyze_functions import *
from multiples_analyze_functions import _get_logg_MS, _list_mag_photometry, _comb_mag_photometry_precomputed

# create a set of simulated triple stars with determined teff/logg/feh that will be analyzed
feh_single_star = 0.0

dir_suffix = ''
process_obj_begin = 0
two_stars = False
equal_teff = False
if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['obj_beg=', 'obj_end=', 'dir_suffix=', 'feh=', 'two_stars=', 'equal_teff='])
    # set parameters, depending on user inputs
    print opts
    for o, a in opts:
        if o == '--obj_beg':
            process_obj_begin = np.int64(a)
        if o == '--obj_end':
            process_obj_end = np.int64(a)
        if o == '--dir_suffix':
            dir_suffix = str(a)
        if o == '--two_stars':
            if np.int64(a) > 0:
                two_stars = True
            else:
                two_stars = False
        if o == '--equal_teff':
            if np.int64(a) > 0:
                equal_teff = True
            else:
                equal_teff = False

s1_fit = True
s2_fit = True
s3_fit = True
teff_comb = list([])
if two_stars:
    teff_possible = np.arange(4800, 6300, 100)[::-1]
    n_teff = len(teff_possible)
    # s3_fit = False
    for i_1 in np.arange(n_teff)[:10]:
        if equal_teff:
            teff_comb.append(np.array([teff_possible[i_1], teff_possible[i_1]]))
        else:
            for i_2 in np.arange(i_1+1, n_teff - 0):
                teff_comb.append(np.array([teff_possible[i_1], teff_possible[i_2]]))
else:
    teff_possible = np.arange(4800, 6300, 100)[::-1]
    n_teff = len(teff_possible)
    # s1_fit = False
    for i_1 in np.arange(n_teff)[:10]:
        if equal_teff:
            teff_comb.append(np.array([teff_possible[i_1], teff_possible[i_1], teff_possible[i_1]]))
        else:
            for i_2 in np.arange(i_1+1, n_teff - 1):
                for i_3 in np.arange(i_2+1, n_teff - 0):
                    teff_comb.append(np.array([teff_possible[i_1], teff_possible[i_2], teff_possible[i_3]]))

print teff_comb
print ' Number of simulations that will be performed:', len(teff_comb)

if 'process_obj_end' not in locals():
    process_obj_end = len(teff_comb)

output_dir = 'Simulation_multiples_fit'+dir_suffix

system('mkdir '+out_dir_root+output_dir)
chdir(out_dir_root+output_dir)

table_out_fits = 'fit_results.fits'
if path.isfile(table_out_fits):
    print 'Reading previous results'
    table_out = Table.read(table_out_fits)
else:
    s1_cols = ['s1_teff1', 's1_feh', 'phot_excs', 'phot_chi2', 's1_sim_p', 'spec_chi2', 's1_sim_f']
    s2_cols = ['s2_teff1', 's2_teff2', 's2_feh', 'phot_excs2', 'phot_chi22', 's2_sim_p', 'spec_chi22', 's2_sim_f']
    s3_cols = ['s3_teff1', 's3_teff2', 's3_teff3', 's3_feh', 'phot_excs3', 'phot_chi23', 's3_sim_p', 'spec_chi23', 's3_sim_f']
    all_cols = np.hstack((s1_cols, s2_cols, s3_cols))
    all_dtype = np.full(len(all_cols), 'float64')
    table_out = Table(names=np.hstack(('sobject_id', 'parallax', all_cols, 'n_stars_p', 'n_stars_f')),
                      dtype=np.hstack(('S40', 'float64', all_dtype, 'int32', 'int32')))

for teff_system in teff_comb[process_obj_begin: process_obj_end]:
    logg_system = _get_logg_MS(teff_system)

    s_id = '_'.join(str(t) for t in teff_system)
    print 'Working on:', s_id

    if np.sum(table_out['sobject_id'] == s_id) >= 1:
        print ' SKIPPING: Already processed'
        continue

    # create object data for this imaginary object
    obj_data = Table(_comb_mag_photometry_precomputed(teff_system, logg_system, feh_single_star), names=p_cols)
    for c in p_cols_sigma:
        obj_data[c] = 0.025
    obj_data['Fe_H_cannon'] = feh_single_star
    obj_data['Teff_cannon'] = np.mean(teff_system)
    obj_data['Logg_cannon'] = np.mean(logg_system)
    obj_data['flag_cannon'] = 0
    obj_data['sobject_id'] = s_id
    obj_data['parallax'] = 100.  # parallax equivalent to distance of 10 pc
    obj_data['r_est'] = 10.  # distance of 10 pc
    obj_data['Fe_H_cannon_orig'] = obj_data['Fe_H_cannon']

    # create a spectrum of this object
    mag_values_3star = _list_mag_photometry(teff_system, logg_system, feh_single_star, p_cols_galah)
    flx_3star = synthetic_spectra_combine(teff_system, logg_system, [feh_single_star], mag_values_3star)
    flx_3star_std = np.full_like(flx_3star, 0.05)
    wvl_3star = cannon_model.dispersion

    sub_dir = str(s_id)
    system('mkdir ' + sub_dir)
    chdir(sub_dir)

    s_time = time()
    # set obj feh to the init value
    fit_res_all = fit_photometry_to_object(obj_data, flx_3star, flx_3star_std, wvl_3star,
                                           fit_single=s1_fit, fit_double=s2_fit, fit_tripple=s3_fit,
                                           # which stellar configuration to fit
                                           complete_wvl_range=True, fe_wvl_range_only=False, write_out=True,
                                           nwalkers=60, n_steps_1=80, n_steps_2=70, n_steps_feh=50, n_threds=30)  # number of fitting steps in MCMC

    # output required processing time
    print 'Fit time: {:.1f} min'.format((time()-s_time)/60.)

    chdir('..')

    # store results to array and write it out
    out_vals = [s_id, 100.]
    for f_r in fit_res_all:
        out_vals.append(f_r)

    # print 'table_out:', len(table_out.colnames)
    # print 'fit_res_all:', len(fit_res_all)
    # print 'out_vals:', len(out_vals)

    table_out.add_row(out_vals)
    table_out.write(table_out_fits, overwrite=True)

