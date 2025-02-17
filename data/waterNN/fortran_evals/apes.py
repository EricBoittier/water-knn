#!/usr/bin/env python3
"""This module defines an ASE interface for any analytical PES
"""
import os
import numpy as np

from warnings import warn
from ase.atoms import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError


class KPoint:
    def __init__(self, s):
        self.s = s
        self.eps_n = []
        self.f_n = []


class APES(FileIOCalculator):
    implemented_properties = ['energy', 'forces']
#    command = './script.sh'
    command = './nnker.x'

    default_parameters = dict()
#    default_parameters = dict(
#        charge=0, mult=1,
#        task='gradient',
#        orcasimpleinput='PBE def2-SVP',
#        orcablocks='%scf maxiter 200 end',
#        )  

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='apes', atoms=None, **kwargs):
        """Construct APES-calculator object."""
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Write coordinates:
        s = "3\n"
        s += "TITLE\n"
        for symbol, xyz in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            s += symbol + " " + str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + "\n"
#        s += "end\n"

        with open('inp.xyz', 'w') as f:
            f.write(s)

        self.atoms = atoms

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        f = open(self.label + '.inp')
        for line in f:
            if line.startswith('geometry'):
                break
        symbols = []
        positions = []
        for line in f:
            if line.startswith('end'):
                break
            words = line.split()
            symbols.append(words[0])
            positions.append([float(word) for word in words[1:]])

#        self.parameters = Parameters.read(self.label + '.ase')
        self.read_results()

    def read_results(self):
        self.read_energy()
        if self.parameters.task.find('gradient') > -1:
            self.read_forces()

    def read_energy(self):
        """Read Energy from output file."""
        text = open('ener.out', 'r').read()
        lines = iter(text.split('\n'))
        # Energy:
        estring = 'FINAL SINGLE POINT ENERGY'
        for line in lines:
            if estring in line:
                energy = float(line.split()[-1])
                break
        self.results['energy'] = energy 

    def read_forces(self):
        """Read Forces from output file."""
        file = open('grad.out', 'r')
        lines = file.readlines()
        file.close()
        getgrad="no"
        Natom=3
        gradients = np.zeros([Natom,3])
        for i, line in enumerate(lines):
            if line.find('# The current gradient') >= 0:
                getgrad="yes";j=0;continue
            if getgrad=="yes" and "#" not in line:
                x, y, z = line.split()
                gradients[j, 0] = float(x)
                gradients[j, 1] = float(y)
                gradients[j, 2] = float(z)
                j += 1
            if '# The end' in line:
                getgrad="no"
        self.results['forces'] = gradients


