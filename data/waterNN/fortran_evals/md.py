#!/usr/bin/env python3
from apes import APES
from ase.atoms import Atoms
from ase.optimize import BFGS
from ase import units
from ase.optimize import FIRE
import ase
import numpy as np
from ase.constraints import FixInternals
from ase.vibrations import Vibrations
from ase.visualize import view
#from ase.vibrations.infrared import InfraRed
import time

import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, ZeroRotation, Stationary
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from os.path import splitext
from ase.md.verlet import VelocityVerlet



calc = APES(task='gradient')
water=ase.io.read('water.xyz')
water.set_calculator(calc)

e_water = water.get_potential_energy()
print(water.get_masses())
#print("Water  energy is", e_water)

#print(water.get_forces())


#Optimization stuff
#energy(water)
opt = BFGS(water)
#opt = FIRE(water)
opt.run(fmax=0.00001)


forces = water.get_forces()
energy = water.get_potential_energy()
positions = water.get_positions()


#assign initial velocities

# Set the momenta corresponding to a temperature T
MaxwellBoltzmannDistribution(water, 300 * units.kB)
Stationary(water)
ZeroRotation(water)

#dyn = Langevin(water, 0.5 * units.fs, 300 * units.kB, 1e-3)
dyn = VelocityVerlet(water, 0.25 * units.fs)


def printenergy(a=water):  # store a reference to water in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
# save the positions of all water after every Xth time step.
traj = Trajectory(str(300)+ 'K_md_water.traj', 'w', water)


start_time = time.time()
# run the dynamics
for i in range(400000):
    dyn.run(1)
    if i%200 == 0:
        epot = water.get_potential_energy() / len(water)
        ekin = water.get_kinetic_energy() / len(water)
        print("Production Step: ", i)
        traj.write(water)
        
        
end_time = time.time()


# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
print("Time per step: ", elapsed_time / args.steps)

