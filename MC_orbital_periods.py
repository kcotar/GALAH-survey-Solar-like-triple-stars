import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un
import astropy.constants as const
from os import system, chdir

plt.rcParams['font.size'] = 15
#np.seterr(divide='ignore', invalid='ignore')

out_dir = 'MC_orbits'
system('mkdir '+out_dir)
chdir(out_dir)

suffix = ''

print '-------------------------'
print '      Outer pair'
print '-------------------------'
# distribution of radial velocity separations - OUTER PAIR
n_systems = 1000000
mean_rv = 0.
M_prim = 2.*const.M_sun  # two Suns in the center
a_dist = (np.random.uniform(100., 350., n_systems) * un.AU).to(un.m)

# linear distribution of semi-major axis
# n_vals = 20000000
# a_vals = np.linspace(100., 350., n_vals)
# p_a_vals = np.linspace(1., 0.3, n_vals)
# p_a_vals = p_a_vals / np.sum(p_a_vals)
# a_dist = (np.random.choice(a_vals, p=p_a_vals, size=n_systems) * un.AU).to(un.m)

# a_const = 10.
# suffix += '_{:.0f}AU'.format(a_const)
#a_dist = (np.random.uniform(a_const, a_const, n_systems) * un.AU).to(un.m)
q_bin = np.random.uniform(0.45, 0.55, n_systems)
sin_i_orb = np.random.uniform(np.sin(np.deg2rad(0.)), np.sin(np.deg2rad(90.)), n_systems)  # rad angle
e_orb = np.random.uniform(0.1, 0.8, n_systems)
# argument of periapsis node rad angle
omeg_orb = np.random.uniform(0., 2.*np.pi, n_systems)
# true anomaly
phase = np.random.uniform(0., 1., n_systems)
M_t = 2*np.pi*phase
E_t = np.full_like(M_t, np.pi)
for i in range(100):
    E_t = M_t + e_orb * np.sin(E_t)
OMEG_orb = 2.*np.arctan(np.sqrt((1.+e_orb)/(1.-e_orb))*np.tan(E_t/2.))

# orbital period of a modeled system
P_orb = np.sqrt(a_dist**3 * 4.*np.pi**2/(const.G*M_prim*(1.+q_bin)))
# determine maximal distance from the mass center for both components
a_dist_1 = a_dist*q_bin/(1.+q_bin)
a_dist_2 = a_dist/(1.+q_bin)
# determine orbital phase factor for both components
orb_phase_1 = np.cos(OMEG_orb+omeg_orb) + e_orb*np.cos(omeg_orb)
orb_phase_2 = np.cos(OMEG_orb+omeg_orb+np.pi) + e_orb*np.cos(omeg_orb+np.pi)  # shifted for 180 deg
# compute both rv values
v_rad_1 = 2.*np.pi*a_dist_1*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_1 + mean_rv
v_rad_2 = 2.*np.pi*a_dist_2*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_2 + mean_rv
# combine them
v_rad_diff = v_rad_2 - v_rad_1

v_rad_sys = 2.*np.pi*a_dist*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_2 + mean_rv

v_rad = v_rad_diff.to(un.km/un.s).value
v_rad_sys = v_rad_sys.to(un.km/un.s).value
# plt.hist(v_rad, bins=100, alpha=0.2, normed=True)

rv_perc = np.percentile(np.abs(v_rad), [68, 95, 99.7])
print 'Period/5 >= 30yr:', 100.*np.sum(P_orb.to(un.yr).value/5. >= 30.)/len(P_orb)
print 'Less than 4:', 100.*np.sum(np.abs(v_rad) <= 4.)/len(v_rad)
print 'Less than 5:', 100.*np.sum(np.abs(v_rad) <= 5.)/len(v_rad)
print 'Less than 6:', 100.*np.sum(np.abs(v_rad) <= 6.)/len(v_rad)
print 'Less than 7:', 100.*np.sum(np.abs(v_rad) <= 7.)/len(v_rad)
print 'Less than 8:', 100.*np.sum(np.abs(v_rad) <= 8.)/len(v_rad)
print 'Less than 9:', 100.*np.sum(np.abs(v_rad) <= 9.)/len(v_rad)

rv_r = 11
plt.figure(figsize=(7, 4))
plt.hist(v_rad, bins=70, range=(-rv_r, rv_r), color='black', alpha=0.2, density=True)
plt.hist(v_rad, bins=70, range=(-rv_r, rv_r), color='black', alpha=1, density=True, histtype='step')
plt.xlim(-rv_r, rv_r)
plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], ['0.0', '', '0.1', '', '0.2', '', '0.3', '', '0.4'])
plt.grid(alpha=0.2, color='black', ls='--')

print 'RV percentiles:', rv_perc

for rv_perc_val in rv_perc:
    plt.axvline(rv_perc_val, ls='--', alpha=0.75, color='red', lw=1.2)
    plt.axvline(-rv_perc_val, ls='--', alpha=0.75, color='red', lw=1.2)

plt.ylabel('Probability density')
plt.xlabel(u'Radial velocity separation v$_2$ - v$_1$ [km s$^{-1}$]')
plt.tight_layout()
# plt.show()
plt.savefig('MC_rv_from_sep'+suffix+'_outer.png', dpi=200)
plt.close()

idx_rv_low = np.abs(v_rad) <= 8
orbits_years = P_orb.to(un.yr).value
print 'Orbits limits:', np.nanmin(orbits_years), np.nanmax(orbits_years), 'years'
plt.figure(figsize=(7, 4))
plt.hist(orbits_years, bins=70, color='black', alpha=0.2, density=True)
plt.hist(orbits_years, bins=70, color='black', alpha=1, density=True, histtype='step')
plt.hist(orbits_years[idx_rv_low], bins=70, color='C3', alpha=0.2, density=True)
plt.ylabel('Probability density')
plt.xlabel(u'Orbital period [yr]')
plt.grid(alpha=0.2, color='black', ls='--')
plt.tight_layout()
plt.savefig('MC_rv_from_sep_period'+suffix+'_outer.png', dpi=200)
plt.close()

print '-------------------------'
print '      Inner pair'
print '-------------------------'
suffix = ''
# distribution of radial velocity separations - INNER PAIR
n_systems = 1000000
mean_rv = 0.
M_prim = 1.*const.M_sun  # two Suns in the center
q_bin = np.random.uniform(0.9, 1.0, n_systems)
sin_i_orb = np.random.uniform(np.sin(np.deg2rad(0.)), np.sin(np.deg2rad(90.)), n_systems)  # rad angle
e_orb = np.random.uniform(0.1, 0.8, n_systems)
# argument of periapsis node rad angle
omeg_orb = np.random.uniform(0., 2.*np.pi, n_systems)
# true anomaly
phase = np.random.uniform(0., 1., n_systems)
M_t = 2*np.pi*phase
E_t = np.full_like(M_t, np.pi)
for i in range(100):
    E_t = M_t + e_orb * np.sin(E_t)
OMEG_orb = 2.*np.arctan(np.sqrt((1.+e_orb)/(1.-e_orb))*np.tan(E_t/2.))

# orbital period of a modeled system
# P_const = 40.
# suffix += '_{:.0f}yr'.format(P_const)
# P_orb = (np.random.uniform(P_const, P_const, n_systems)*un.yr).to(un.s)
P_orb = P_orb/5.
a_dist = ((const.G * M_prim * (1. + q_bin) * P_orb**2) / (4. * np.pi**2))**(1./3.)

# determine maximal distance from the mass center for both components
a_dist_1 = a_dist*q_bin/(1.+q_bin)
a_dist_2 = a_dist/(1.+q_bin)
# determine orbital phase factor for both components
orb_phase_1 = np.cos(OMEG_orb+omeg_orb) + e_orb*np.cos(omeg_orb)
orb_phase_2 = np.cos(OMEG_orb+omeg_orb+np.pi) + e_orb*np.cos(omeg_orb+np.pi)  # shifted for 180 deg
# compute both rv values
v_rad_1 = 2.*np.pi*a_dist_1*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_1 + mean_rv
v_rad_2 = 2.*np.pi*a_dist_2*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_2 + mean_rv
# combine them
v_rad_diff = v_rad_2 - v_rad_1

v_rad_sys = 2.*np.pi*a_dist*sin_i_orb/(P_orb*np.sqrt(1.-e_orb**2))*orb_phase_2 + mean_rv

v_rad = v_rad_diff.to(un.km/un.s).value
v_rad_sys = v_rad_sys.to(un.km/un.s).value

rv_perc = np.percentile(np.abs(v_rad), [68, 95, 99.7])
print 'Less than 3:', 100.*np.sum(np.abs(v_rad) <= 3)/len(v_rad)
print 'Less than 4:', 100.*np.sum(np.abs(v_rad) <= 4)/len(v_rad)
print 'Less than 5:', 100.*np.sum(np.abs(v_rad) <= 5)/len(v_rad)
print 'Less than 6:', 100.*np.sum(np.abs(v_rad) <= 6)/len(v_rad)
print 'Less than 7:', 100.*np.sum(np.abs(v_rad) <= 7)/len(v_rad)
print 'Less than 8:', 100.*np.sum(np.abs(v_rad) <= 8)/len(v_rad)

rv_r = 11
plt.figure(figsize=(7, 4))
plt.hist(v_rad, bins=70, range=(-rv_r, rv_r), color='black', alpha=0.2, density=True)
plt.hist(v_rad, bins=70, range=(-rv_r, rv_r), color='black', alpha=1, density=True, histtype='step')
plt.xlim(-rv_r, rv_r)
plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], ['0.0', '', '0.1', '', '0.2', '', '0.3'])
plt.grid(alpha=0.2, color='black', ls='--')

print 'RV percentiles:', rv_perc

for rv_perc_val in rv_perc:
    plt.axvline(rv_perc_val, ls='--', alpha=0.75, color='red', lw=1.2)
    plt.axvline(-rv_perc_val, ls='--', alpha=0.75, color='red', lw=1.2)

plt.ylabel('Probability density')
plt.xlabel(u'Radial velocity separation v$_2$ - v$_1$ [km s$^{-1}$]')
plt.tight_layout()
# plt.show()
plt.savefig('MC_rv_from_sep'+suffix+'_inner.png', dpi=200)
plt.close()

idx_rv_low = np.abs(v_rad) <= 5
orbits_years = P_orb.to(un.yr).value
print 'Orbits limits:', np.nanmin(orbits_years), np.nanmax(orbits_years), 'years'
plt.figure(figsize=(7, 4))
plt.hist(orbits_years, bins=70, color='black', alpha=0.2, density=True)
plt.hist(orbits_years, bins=70, color='black', alpha=1, density=True, histtype='step')
plt.hist(orbits_years[idx_rv_low], bins=70, color='C3', alpha=0.2, density=True)
plt.ylabel('Probability density')
plt.xlabel(u'Orbital period [yr]')
plt.grid(alpha=0.2, color='black', ls='--')
plt.tight_layout()
plt.savefig('MC_rv_from_sep_period'+suffix+'_inner.png', dpi=200)
plt.close()

semi_axis = a_dist.to(un.AU).value
plt.figure(figsize=(7, 4))
plt.hist(semi_axis, bins=70, color='black', alpha=0.2, density=True)
plt.hist(semi_axis, bins=70, color='black', alpha=1, density=True, histtype='step')
plt.ylabel('Probability density')
plt.xlabel(u'Semi-major axis [AU]')
plt.grid(alpha=0.2, color='black', ls='--')
plt.tight_layout()
plt.savefig('MC_rv_from_sep_semiaxis'+suffix+'_inner.png', dpi=200)
plt.close()

