# The GALAH survey: unresolved triple Sun-like stars discovered by the Gaia mission 

The paper was published at [MNRAS](https://doi.org/10.1093/mnras/stz1397).

Code used in the analysis of Solar twin stars multiplicity and production of the acompanying science paper that describes identification and characterization of triple system candidates.

### Detection od Solar-twin candidates

Done in a seprate repository/science paper.

### Creation of photometric one star model

Steps:
* Combine observations of multiple phomoteric surveys and Gaia astrometry - `prepare_data.py`
* Add reddeding and extinction information to the table - `prepare_data_reddening.py`
* Prepare photometric table with median photometric informations - `prepare_photometry_reference_table.py`

### Creation of spectroscopic one star model

To train the spectroscopic Cannon model, sript `Cannon_model_train.py` was used.

### Fit multiple one star models to the observation and determine systems configuration

The core of the analysis is made by function and procedures writen in `multiples_analyze_functions.py`. The proceured is run from `multiples_analyze.py`, where all relavant parameters for the fitting procedure are set and determined.

## Orbital characterization and simulations

Monte Carlo analysis of orbital periods is presented in the script `MC_orbital_periods.py`.

### Test and simulations

Multiple test ans imulations were run to ensure the quality of the results:
* Observational offsets using Galaxia galactic model - `galaxia_determine_observationa_offsets.py`
* Analysis of synthetic multiple systems - `multiples_analyze_binarysystem_simulation.py`
