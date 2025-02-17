#!/usr/bin/env python3
from apes import APES
from ase.atoms import Atoms
from ase.optimize import BFGS
from ase.optimize import FIRE
import ase
import numpy as np
from ase.constraints import FixInternals
from ase.vibrations import Vibrations
from ase.visualize import view
#from ase.vibrations.infrared import InfraRed

calc = APES(task='gradient')
water=ase.io.read('water.xyz')
water.set_calculator(calc)

e_water = water.get_potential_energy()
print(water.get_masses())
#print("Water  energy is", e_water)

#print(water.get_forces())


#Optimization stuff
opt = BFGS(water)
#opt = FIRE(water)
opt.run(fmax=0.00001)


forces = water.get_forces()
energy = water.get_potential_energy()
positions = water.get_positions()


print(energy, "eV", energy * 23.0605419, "kcal/mol")
print("Final positions are written in test.xyz file (in angstrom)")
ase.io.write('opt_water.xyz', water)


vib = Vibrations(water,delta=0.01)
vib.run()
vib.summary()
vib.clean()
