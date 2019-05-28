import numpy as np
from astropy.table import Table, join, unique, vstack, unique
import astropy.units as un
import astropy.coordinates as coord
from Reddening_test import stilism as st
from dustmaps.bayestar import BayestarWebQuery, fetch, BayestarQuery
from joblib import Parallel, delayed
from astroquery.gaia import Gaia
from os import path

# --------------------------------------------------------
# ---------------- Data inputs and reading ---------------
# --------------------------------------------------------

galah_dir = '/shared/ebla/cotar/'

galah_photo_dir = galah_dir + 'photometry/'

# kinematics data
print 'Reading Gaia data'
galah_gaia = Table.read(galah_dir+'sobject_iraf_53_gaia_ruwe.fits')['ra', 'dec', 'sobject_id','source_id','parallax','parallax_error','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag', 'ruwe']

# get Bayesian distances
i_min = 0
i_step = 15000
bayesian_fits = galah_dir+'sobject_iraf_53_gaia_distance.fits'
if path.isfile(bayesian_fits):
    print 'Reading infered distances'
    bayesian_distances_all = Table.read(bayesian_fits)
else:
    bayesian_distances_all = list([])
    u_sources = np.unique(galah_gaia['source_id'])
    while i_min < len(u_sources):
        print i_min
        sids = ','.join([str(s) for s in u_sources[i_min:i_min+i_step]])
        q = 'SELECT * FROM external.gaiadr2_geometric_distance WHERE source_id in ('+sids+')'
        gaia_job = Gaia.launch_job_async(q, dump_to_file=False)
        bayesian_distances_all.append(gaia_job.get_data())
        i_min += i_step
    bayesian_distances_all = vstack(bayesian_distances_all)
    # remove units
    for col in bayesian_distances_all.colnames:
        bayesian_distances_all[col].unit = None
    bayesian_distances_all.write(bayesian_fits, overwrite=True)

date_string = '20180327'
# spectroscopic data
print 'Reading Galah'
# galah_data = Table.read(galah_dir+'sobject_iraf_iDR2_180325_cannon.fits')['sobject_id', 'Teff_cannon', 'e_Teff_cannon', 'Fe_H_cannon', 'e_Fe_H_cannon', 'Logg_cannon', 'e_Logg_cannon', 'flag_cannon', 'red_flag', 'Vsini_cannon']
galah_data = Table.read(galah_dir+'GALAH_iDR3_v1_181221_cannon.fits')['sobject_id', 'Teff_cannon', 'e_Teff_cannon', 'Fe_H_cannon', 'e_Fe_H_cannon', 'Logg_cannon', 'e_Logg_cannon', 'flag_cannon', 'red_flag', 'Vsini_cannon']
# galah_cannon_data = galah_cannon_data[galah_cannon_data['flag_cannon'] == 0]
# galah_cannon_data.remove_column('flag_cannon')
# photometric data
print 'Reading photometry'
galah_apass = Table.read(galah_photo_dir+'apass_dr53_'+date_string+'.csv')['sobject_id','Vmag','e_Vmag','Bmag','e_Bmag','gpmag','e_gpmag','rpmag','e_rpmag','ipmag','e_ipmag']
galah_wise = Table.read(galah_photo_dir+'wise_dr53_'+date_string+'.csv')['sobject_id','W1mag','W2mag','e_W1mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag']
galah_2mass = Table.read(galah_photo_dir+'2mass_dr53_'+date_string+'.csv')['sobject_id','Jmag','Hmag','Kmag','e_Jmag','e_Hmag','e_Kmag']
galah_wise = unique(galah_wise, keys='sobject_id', keep='first')
galah_2mass = unique(galah_2mass, keys='sobject_id', keep='first')
galah_apass = unique(galah_apass, keys='sobject_id', keep='first')
# aditional photometry data
galah_panstars = Table.read(galah_photo_dir+'panstarrs_dr53_'+date_string+'.csv')['sobject_id', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'ymag', 'e_ymag']
galah_panstars = unique(galah_panstars, keys='sobject_id', keep='first')
# galah_galex = Table.read(galah_photo_dir+'galex_dr53_'+date_string+'.csv')['sobject_id','fuv','nuv']
# galah_galex = unique(galah_galex, keys='sobject_id', keep='first')

print 'Joining datasets'
galah_data_complete = join(galah_data, galah_apass, keys='sobject_id', join_type='left')
galah_data_complete = join(galah_data_complete, galah_wise, keys='sobject_id', join_type='left')
galah_data_complete = join(galah_data_complete, galah_2mass, keys='sobject_id', join_type='left')
galah_data_complete = join(galah_data_complete, galah_gaia, keys='sobject_id', join_type='left')
# aditional photometry data
galah_data_complete = join(galah_data_complete, galah_panstars, keys='sobject_id', join_type='left')
# galah_data_complete = join(galah_data_complete, galah_galex, keys='sobject_id', join_type='left')
print galah_data_complete['sobject_id', 'source_id', 'parallax']
galah_data_complete = join(galah_data_complete, bayesian_distances_all['source_id', 'r_est', 'r_lo', 'r_hi'], keys='source_id', join_type='left')
galah_data_complete = galah_data_complete[np.argsort(galah_data_complete['sobject_id'])]

# add null/nan
idx_missing = galah_data_complete['source_id'] < 0
for c in ['r_est', 'r_lo', 'r_hi']:
    galah_data_complete[c][idx_missing] = np.nan
print galah_data_complete['sobject_id', 'source_id', 'parallax', 'r_est']

# get E(B-V) values from multiple sources
print ' Bayestar E(B-V)'
coords = coord.SkyCoord(galah_data_complete['ra']*un.deg,
                        galah_data_complete['dec']*un.deg,
                        # distance=1e3/galah_data_complete['parallax']*un.pc,
                        distance=galah_data_complete['r_est']*un.pc,
                        frame='icrs')

bayestar = BayestarWebQuery(version='bayestar2017')
ebv = bayestar(coords, mode='median')
galah_data_complete['ebv_B'] = ebv


# print ' Stilism E(B-V)'
# ebv2 = list([])
# for i_s, star_data in enumerate(galah_data_complete):
def get_ebv_stilism(i_s):
    star_data = galah_data_complete[i_s]
    if i_s % 1000 == 0:
        print i_s
    if not np.isfinite(star_data['r_est']):
        # ebv2.append(np.nan)
        # continue
        return np.nan
    stilism_data = st.reddening(star_data['ra'], un.deg, star_data['dec'], un.deg, 'icrs')
    # idx_close = np.argmin(np.abs(stilism_data[0] - 1e3/star_data['parallax']))
    idx_close = np.argmin(np.abs(stilism_data[0] - star_data['r_est']))
    # ebv2.append(stilism_data[1][idx_close])
    return stilism_data[1][idx_close]


print ' Stilism E(B-V)'
ebv2 = Parallel(n_jobs=20)(delayed(get_ebv_stilism)(i_row) for i_row in range(len(galah_data_complete)))
galah_data_complete['ebv_S'] = ebv2

s_ids = galah_data_complete['sobject_id']
galah_data_complete.remove_columns(['sobject_id','ra','dec'])
# galah_data_complete = galah_data_complete.filled(np.nan)

galah_data_complete.add_column(s_ids)
print len(galah_data_complete)

# # use E(B-V) to compute extinction in used photometric bands
# R = {'gmag':3.384, 'rmag':2.483, 'imag':1.838, 'zmag':1.414, 'ymag':1.126,
#      'Jmag':0.650, 'Hmag':0.327, 'Kmag':0.166,
#      'Vmag':2.742, 'Bmag':3.626, 'gpmag':3.384, 'rpmag':2.483, 'ipmag':1.838,
#      'phot_g_mean_mag':2.742, 'phot_bp_mean_mag':3.626, 'phot_rp_mean_mag':2.169}  # using Landot V, B, R# use E(B-V) to compute extinction in used photometric bands
R = {'gmag':3.172, 'rmag':2.271, 'imag':1.682, 'zmag':1.322, 'ymag':1.087,
     'Jmag':0.709, 'Hmag':0.449, 'Kmag':0.302,
     'Vmag':2.742, 'Bmag':3.626, 'gpmag':3.303, 'rpmag':2.285, 'ipmag':1.698,
     'phot_g_mean_mag':2.742, 'phot_bp_mean_mag':3.626, 'phot_rp_mean_mag':2.169}  # using Landot V, B, R

# correct magnitude values for the given extinction coefficients
for p_band in R.keys():
    galah_data_complete[p_band] = galah_data_complete[p_band] - galah_data_complete['ebv_S'] * R[p_band]

print 'Saving data'
date_string = '20181221'
galah_data_complete.write(galah_dir+'galah_cannon_DR3_gaia_photometry_'+date_string+'_ebv-corr.fits', overwrite=True)
