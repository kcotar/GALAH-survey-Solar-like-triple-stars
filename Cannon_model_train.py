import imp
import matplotlib.pyplot as plt
from astropy.table import Table, join, unique
import thecannon as tc
import numpy as np

imp.load_source('spectra_collection_functions', '../Carbon-Spectra/spectra_collection_functions.py')
from spectra_collection_functions import read_pkl_spectra, save_pkl_spectra, CollectionParameters


# --------------------------------------------------------
# ---------------- Read data -----------------------------
# --------------------------------------------------------
print 'Reading GALAH parameters'
galah_data_dir = '/shared/ebla/cotar/'
date_string = '20180327'
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')
# cannon_data = Table.read(galah_data_dir + 'sobject_iraf_iDR2_180325_cannon.fits')
cannon_data = Table.read(galah_data_dir + 'GALAH_iDR3_v1_181221_cannon.fits')
# cannon_data = Table.read(galah_data_dir + 'sobject_iraf_iDR2_180108_sme.fits')
cannon_data = unique(cannon_data, keys='sobject_id')
general_data = join(general_data, cannon_data['sobject_id', 'Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'flag_cannon', 'Vsini_cannon', 'Vmic_cannon'], keys='sobject_id', join_type='left')
# general_data = join(general_data, cannon_data['sobject_id', 'Teff_sme', 'Logg_sme', 'Feh_sme', 'flag_cannon', 'Vmic_sme', 'Vsini_sme'], keys='sobject_id', join_type='left')

idx_ok = general_data['red_flag'] == 0  # remove flats, reduction and wavelength problems
idx_ok = np.logical_and(idx_ok, np.isfinite(general_data['Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'Vsini_cannon'].to_pandas().values).all(axis=1))
idx_ok = np.logical_and(idx_ok, general_data['flag_cannon'] == 0)
idx_ok = np.logical_and(idx_ok, general_data['snr_c2_iraf'] >= 20.)
idx_ok = np.logical_and(idx_ok, general_data['Vsini_cannon'] <= 25.)
idx_ok = np.logical_and(idx_ok, general_data['Vmic_cannon'] <= 1.8)
idx_ok = np.logical_and(idx_ok, general_data['Fe_H_cannon'] >= -1.5)

# compute logg from the teff in the case of dwarf star
y_logg_thr = -4e-4 * general_data['Teff_cannon'] + 5.85
idx_ok = np.logical_and(idx_ok, general_data['Logg_cannon'] >= y_logg_thr)

sobject_observe = general_data['sobject_id'][idx_ok]
# sobject_observe = cannon_data['sobject_id']

idx_rows_read = np.where(np.in1d(general_data['sobject_id'], sobject_observe))[0]

spectra_file_list = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd2_5640_5880_wvlstep_0.050_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl',
                     'galah_dr53_ccd4_7700_7895_wvlstep_0.070_ext4_'+date_string+'.pkl']

min_wvl = np.array([4720, 5660, 6480, 7700])
max_wvl = np.array([4900, 5870, 6730, 7880])

# as used by Sven
# min_wvl = np.array([4715.94, 5650.06, 6480.52, 7693.50])
# max_wvl = np.array([4896.00, 5868.25, 6733.92, 7875.55])

print 'Number of object to be fitted:', len(idx_rows_read)

spectral_data = list([])
wvl_data = list([])
for i_b in [0, 1, 2, 3]:  # [0, 1, 2, 3]:
    # parse interpolation and averaging settings from filename
    csv_param = CollectionParameters(spectra_file_list[i_b])
    ccd = csv_param.get_ccd()
    wvl_values = csv_param.get_wvl_values()

    # determine wvls that will be read from the spectra
    idx_wvl_read = np.where(np.logical_and(wvl_values >= min_wvl[i_b], wvl_values <= max_wvl[i_b]))[0]
    wvl_values = wvl_values[idx_wvl_read]
    wvl_data.append(wvl_values)

    # read limited number of columns instead of full spectral dataset
    print 'Reading resampled/interpolated GALAH spectra - band', i_b+1
    spectral_data.append(read_pkl_spectra(galah_data_dir + spectra_file_list[i_b],
                                          read_cols=idx_wvl_read, read_rows=idx_rows_read))

spectral_data = np.hstack(spectral_data)
wvl_data = np.hstack(wvl_data)
print spectral_data.shape

# nan values handling
idx_nan = ~ np.isfinite(spectral_data)
n_nan = np.sum(idx_nan)
if n_nan > 0:
    print 'Correcting '+str(n_nan)+' nan values'
    spectral_data[idx_nan] = 1.

# negative values handling
idx_neg = spectral_data < 0.
if np.sum(idx_neg) > 0:
    spectral_data[idx_neg] = 0.
# large positive values handling
idx_gros = spectral_data > 1.2
if np.sum(idx_gros) > 0:
    spectral_data[idx_gros] = 1.2

# run Cannon learning procedure
# Load the table containing the training set labels, and the spectra.
list_cols_fit = ['Teff_cannon', 'Logg_cannon', 'Fe_H_cannon', 'Vsini_cannon']
training_set = general_data[list_cols_fit][idx_rows_read]

normalized_ivar = np.full_like(spectral_data, 1/0.02**2)
# Create the model that will run in parallel using all available cores.

vectorizer = tc.vectorizer.polynomial.PolynomialVectorizer(label_names=list_cols_fit, order=2)
model = tc.CannonModel(training_set, spectral_data, normalized_ivar, vectorizer, dispersion=wvl_data)

# Train the model!
print 'Model training'
model.train(threads=10)

model.write('model_cannon181221_DR3_ccd1234_noflat_red0_cannon0_oksnr_vsiniparam_dwarfs_002.dat', include_training_set_spectra=False, overwrite=True, protocol=-1)
