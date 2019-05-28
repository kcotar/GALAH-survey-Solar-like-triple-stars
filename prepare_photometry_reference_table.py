from multiples_analyze_functions import *

# more data filtering
# cannon_flag === 0 also already incorporated into median_photometry function
idx_cannon_ok = galah_gaia_data['flag_cannon'] == 0
# idx_parallax_ok = galah_gaia_data['parallax_error'] < galah_gaia_data['parallax']*0.2
# idx_parallax_ok = galah_gaia_data['r_est'] > 0
idx_parallax_ok = galah_gaia_data['ruwe'] <= 1.4
idx_ok = np.logical_and(idx_ok,
                        np.logical_and(idx_cannon_ok, idx_parallax_ok))
print 'idx_ok:', np.sum(idx_ok)
fits_file = 'median_photometry_cannon0_DR3_ruwe_80_005_01_all_noebv.fits'

cols_all = np.hstack((['teff', 'logg', 'feh'], p_cols))
out_data = Table(names=cols_all,
                 dtype=np.repeat('float64', len(cols_all)))

for teff in np.arange(4400., 6600., 25.):
    print teff
    for logg in np.arange(3.5, 5.5, 0.025):
        for feh in np.arange(-1., 1., 0.025):
            phot_median = median_photometry(teff, logg, feh, p_cols,
                                            d_teff=80, d_logg=0.05, d_feh=0.1,
                                            plot_res=False, min_data=15, idx_init=idx_ok)
            if len(phot_median) == 0:
                continue

            out_data.add_row(np.hstack((teff, logg, feh, phot_median)))

out_data.write(fits_file, overwrite=True)
