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

