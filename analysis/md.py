from ff_energy.simulations import charmm, plots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import patchworklib as pw
from ase.visualize import view
import MDAnalysis as mda
from MDAnalysis.coordinates.XYZ import XYZWriter as XYZWriter
from ase import Atoms
import numpy as np
from pint import UnitRegistry
import warnings
# ignore the casting errors for units
warnings.simplefilter("ignore")
ureg = UnitRegistry()

#  constants
Avogadro_const = 6.02214129 * 10**23 * ureg("mol^-1")  # % mol-1

# def get_density(volume, N_res, MW):
#     """
#     get the density of the simulation for a given volume, number of molecules and
#     molecular weight :param volume: m^3 :param N_res: number :param MW: g/mol
#     :return: density in g/m^3
#     """
#     MW = MW * ureg("g/mol")

#     return (N_res * MW) / (Avogadro_const * volume)

# def get_dG_(total_energy, nwater=2000, usingle = -228.20148):
#     tot = total_energy.mean()
#     Ubox = tot/ nwater
#     # Ubox = -481858.7536643403 / 2000
#     Ubox = Ubox * ureg("kcal/mol")
#     Usingle = usingle * ureg("kcal/mol")
#     Gas_const = 8.3144621 * ureg("J/(mol*K)")  # % JK^−1mol^−1
#     Gas_const = Gas_const.to("kcal/(mol*K)")
#     T = 298.0 * ureg("K")
#     dHvap = Usingle - Ubox + T*Gas_const
#     return dHvap.magnitude

# def get_kappa_(volume, T):
#     """
#     Calculate isothermal compressibility
#     """
#     mean_volume = volume.mean()
#     mean_volume =  mean_volume * ureg("angstrom^3")
#     mean_sqr_volume = (volume**2).mean()
#     mean_sqr_volume = mean_sqr_volume * ureg("(angstrom^3)**2")
#     T = T * ureg("K")
#     Vavg2 = mean_volume * mean_volume
#     denom = (T * Boltzmann_const * mean_volume)
#     nom = (mean_sqr_volume - Vavg2)
#     kappa = nom / denom
#     return kappa.to("1/atm") * 10**6

# def print_vol_error(cl):
#     cl["prod"] = cl[""]
#     v = cl[cl["prod"]].volume.mean() * ureg("angstrom^3")
#     exp_dens = 0.99669 * ureg("g/cm^3")
#     dens = get_density(v, 2000, 18).to("g/cm^3") #- 0.997 * ureg("g/cm^3")
#     error = dens - exp_dens
#     return dens


# def get_job_data(files):
#     densities = []
#     cls_ = []
#     jobids = []
#     for _ in files:
#         # try:
#         print(_)
#         cl = charmm.read_charmm_log(_)
#         dens = print_vol_error(cl)
#         cl["dens"] = dens
#         cls_.append(cl)
#         jobids.append(_)
#         # except Exception as e:
#         #     print(e)
#     return {k: v for k,v in zip(jobids, cls_)}

def print_vol_error(cl):
    cl["prod"] = cl["dcd"].apply(lambda x: "dyna" in str(x))
    v = cl[cl["prod"]].volume.mean() * ureg("angstrom^3")
    dens = get_density(v, 2000, 18).to("g/cm^3") #- 0.997 * ureg("g/cm^3")
    return dens

def get_dG_(cl, nwater=2000, usingle = -228.57195772693333):
    cl["prod"] = cl["dcd"].apply(lambda x: "dyna" in str(x))
    total_energy = cl[cl["prod"]].tot
    tot = total_energy.mean()
    Ubox = tot/ nwater
    # Ubox = -481858.7536643403 / 2000
    Ubox = Ubox * ureg("kcal/mol")
    Usingle = usingle * ureg("kcal/mol")
    Gas_const = 8.3144621 * ureg("J/(mol*K)")  # % JK^−1mol^−1
    Gas_const = Gas_const.to("kcal/(mol*K)")
    T = 298.0 * ureg("K")
    dHvap = Usingle - Ubox + T*Gas_const
    return dHvap

def get_job_data(files):
    densities = []
    cls_ = []
    jobids = []
    for _ in files:
        # try:
        print(_)
        cl = charmm.read_charmm_log(_)
        dens = print_vol_error(cl)
        cl["dens"] = dens
        cls_.append(cl)
        jobids.append(_)
        # except Exception as e:
        #     print(e)
    return {k: v for k,v in zip(jobids, cls_)}

import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from pathlib import Path
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def load_unwrapped_dcd(base):
    psfpath = base / "water.psf"
    dcdpath = base / "dcd" / "unwrapped.dcd"
    u = mda.Universe(psfpath, dcdpath)
    return u

def calc_msd(u):
    MSD = msd.EinsteinMSD(u, select='type OT', msd_type='xyz', fft=True)
    NSKIP = 1
    MSD.run(0, -1, NSKIP)
    return MSD

def fit_D_from_MSD(MSD):
    msd_results =  MSD.results.timeseries
    nframes = MSD.n_frames
    timestep = NSKIP * 10_000 * 0.0002 # this needs to be the actual time between frames
    lagtimes = np.arange(nframes)*timestep # make the lag-time axis

    start_time = 0
    start_index = int(start_time/timestep)
    end_index = -1

    linear_model = linregress(
        lagtimes[start_index:end_index],
        msd_results[start_index:end_index]
    )
    slope = linear_model.slope
    error = linear_model.rvalue
    D = slope * 1/(2*MSD.dim_fac)
    return {"lagtimes": lagtimes, "msd": msd_results, "D": D}

def plot_self_diffusion(results):
    
    lagtimes = results["lagtimes"]
    msd_results = results["msd"]
    D = results["D"]
    
    exact = lagtimes*6
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(lagtimes, msd_results, c="black", ls="-", label=r'Observed')
    plt.plot(lagtimes[start_index:], 
             lagtimes[start_index:]*slope + linear_model.intercept, c="r", label="Fit")
    plt.title(f"$D = ${D / 0.1 :.2f}")
    ax.plot(lagtimes, exact, c="black", ls="--", label=r'3D Brownian Motion')
    plt.legend()
    plt.xlabel("Time [ps]", fontsize=20)
    plt.ylabel("MSD [$\mathrm{\AA}^{3}$]", fontsize=20)
    # plt.savefig("msd_water_nn.pdf")
    # plt.show()