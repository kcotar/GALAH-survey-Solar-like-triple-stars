import imp, os
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
from time import time
from astropy.table import Table, vstack, Column


sim_dir = '/home/klemen/GALAH_data_depricated/Galaxia_simulation/'
sim_dir += 'GALAH/'
# sim_file = 'galaxy_galah_complete_mag_10_16.fits'
sim_file = 'galaxy_galah_complete_mag_10_16_fields_r1.0.fits'

print 'Reading Galaxia survey data'
sim_data = Table.read(sim_dir + sim_file)
# aditional GALAH and solar limit(s)
# sim_data = sim_data[np.abs(sim_data['glat']) > 10]
# sim_data = sim_data[np.abs(sim_data['feh']) < 0.05]

# plt.scatter(sim_data['glon'], sim_data['glat'], lw=0, s=0.2, alpha=0.01)
# plt.show()
# plt.close()

sun_u = 5.61
sun_b = 5.44
sun_v = 4.81
d_mag = 0.05

print 'Looking at magnitudes'
# filter absolute magnitudes
idx_absmag = np.logical_and(np.abs(sim_data['ubv_v'] - sun_v) <= d_mag,
                            np.abs((sim_data['ubv_b'] - sim_data['ubv_v']) - (sun_b - sun_v)) <= d_mag)

# compute apparent magnitude
dist_pc = 1e3 * np.sqrt(sim_data['px']**2 + sim_data['py']**2 + sim_data['pz']**2)
sim_data['ubv_v_app'] = sim_data['ubv_v'] + 5.*(np.log10(dist_pc) - 1.)  # + sim_data['exbv_solar']*0.53

min_mag = 12
max_mag = 14
idx_appmag_single = np.logical_and(sim_data['ubv_v_app'] >= min_mag, sim_data['ubv_v_app'] <= max_mag)
mag_offset = np.log(2.)/np.log(2.5)
idx_appmag_double = np.logical_and(sim_data['ubv_v_app'] >= min_mag+mag_offset, sim_data['ubv_v_app'] <= max_mag+mag_offset)
mag_offset = np.log(3.)/np.log(2.5)
idx_appmag_triple = np.logical_and(sim_data['ubv_v_app'] >= min_mag+mag_offset, sim_data['ubv_v_app'] <= max_mag+mag_offset)

n_1 = np.sum(np.logical_and(idx_absmag, idx_appmag_single))
n_2 = np.sum(np.logical_and(idx_absmag, idx_appmag_double))
n_3 = np.sum(np.logical_and(idx_absmag, idx_appmag_triple))

print sim_data[idx_absmag]

print np.log([2, 3, 4])/np.log(2.5)
print n_1, n_2, n_3
print 1.*n_1/n_1, 1.*n_2/n_1, 1.*n_3/n_1
