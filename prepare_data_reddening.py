import numpy as np
from astropy.table import Table
import astropy.units as un
import astropy.coordinates as coord
from Reddening_test import stilism as st
from dustmaps.bayestar import BayestarWebQuery
from joblib import Parallel, delayed

# --------------------------------------------------------
# ---------------- Data inputs and reading ---------------
# --------------------------------------------------------

galah_dir = '/shared/ebla/cotar/Gaia_DR2_RV/'

# kinematics data
print 'Reading Gaia data'
galah_gaia = Table.read(galah_dir+'Gaia_mag16_random50000000.fits')
for col in galah_gaia.colnames:
    galah_gaia[col].unit = None

print len(galah_gaia), np.sum(galah_gaia['parallax'] >= 0)
galah_gaia['parallax'][galah_gaia['parallax'] < 0.] = 1e3/10000.

# get E(B-V) values from multiple sources
print ' Bayestar E(B-V)'
galah_gaia['ebv_B'] = np.nan
i_s = 0
i_n = 200000
while i_s < len(galah_gaia):
    i_e = i_s + i_n
    print i_s, i_e
    coords = coord.SkyCoord(galah_gaia['ra'][i_s:i_e]*un.deg,
                            galah_gaia['dec'][i_s:i_e]*un.deg,
                            distance=1e3/galah_gaia['parallax'][i_s:i_e]*un.pc,
                            frame='icrs')

    bayestar = BayestarWebQuery(version='bayestar2017')
    ebv = bayestar(coords, mode='median')
    galah_gaia['ebv_B'][i_s:i_e] = ebv
    i_s += i_n

print galah_gaia


def get_ebv_stilism(i_s):
    star_data = galah_gaia[i_s]
    if i_s % 2000 == 0:
        print i_s
    if not np.isfinite(star_data['parallax']) and star_data['parallax'] < 0:
        return np.nan
    stilism_data = st.reddening(star_data['ra'], un.deg, star_data['dec'], un.deg, 'icrs')
    idx_close = np.argmin(np.abs(stilism_data[0] - 1e3/star_data['parallax']))
    return stilism_data[1][idx_close]


print ' Stilism E(B-V)'
ebv2 = Parallel(n_jobs=70)(delayed(get_ebv_stilism)(i_row) for i_row in range(len(galah_gaia)))
galah_gaia['ebv_S'] = ebv2

print 'Saving data'
galah_gaia.write(galah_dir+'Gaia_mag16_random50000000_reddening.fits', overwrite=True)
