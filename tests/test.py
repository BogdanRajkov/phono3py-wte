from __future__ import annotations

from pathlib import Path

import phonopy
import pytest
from phonopy import Phonopy
import numpy as np
import phono3py
from phono3py import Phono3py
from wte.velocity_operator import VelocityOperator
from phonopy.physical_units import get_physical_units

np.set_printoptions(linewidth=120)
data = Path(__file__).parent / "data"

#load data 
ph_data = phonopy.load(data / "copper-phonopy.yaml",force_constants_filename=data / "copper-fc2.hdf5",is_compact_fc=False)
# ph_data = phonopy.load(data / "si-data/phono3py_disp.yaml",force_constants_filename=data / "si-data/fc2.hdf5",is_compact_fc=False)

# qpoint in crystal coordinates
# qpoint = [[0.02, 0.0, 0.0]]
# qpoint = [[0.1, 0.22, 0.33]]
qpoint = [[0.22, 0.1, 0.33]]
# qpoint = [ 0.42339524,  0.19758444, -0.00940878]

# xs = np.linspace(0.5, 0.5, 1, endpoint=True)
# qpoints = np.zeros((1,) + xs.shape + (3,), dtype=float)
# qpoints[0, :, 0] = xs
# qpoints[0, :, 2] = xs

# overrides above!
qpoints = np.empty((1, 1, 3))
qpoints[0] = qpoint
print(qpoints)

evals = np.zeros(xs.shape + (3,), dtype=float)

print(qpoints.shape)
ph_data.run_band_structure(qpoints)
f = ph_data.get_band_structure_dict()['frequencies']
print(f[0] * get_physical_units().THzToEv * 1000)

# for it, qpoint in enumerate(qpoints):
#     assert ph_data.dynamical_matrix is not None
#     ph_data.dynamical_matrix.run(qpoint)
#     dm = ph_data.dynamical_matrix.dynamical_matrix
#     assert dm is not None
#     evals[it] = np.linalg.eigh(dm)[0]

# generate eigenvectors and eigenvalues
# evals *= get_physical_units().DefaultToTHz
# print(evals[1:, 2] / evals[1:, 0])


ph_units = get_physical_units()
calc_units = phonopy.physical_units.get_calculator_physical_units(ph_data.calculator)
velocity_factor = ph_units.THz * calc_units.distance_to_A * ph_units.Angstrom

# generate velocity op at the same point
v_ops = VelocityOperator(
    ph_data.dynamical_matrix, symmetry=ph_data.primitive_symmetry, frequency_factor_to_THz=ph_data.unit_conversion_factor
)
v_ops.run(qpoint)
# looks like this outputs something in velocity_operators[q_idx][b1, b2, direction]
v_op = v_ops.velocity_operators[0]
sq_mod = np.einsum("bBx,Bbx->bBx",v_op, v_op).real * velocity_factor**2

np.savetxt('sq_mod', sq_mod.transpose((2, 0, 1)).reshape((-1, 3)))
for i in range(3):
    np.savetxt(f'sq_mod_{i}.log', sq_mod[..., i], fmt='%.4e')

# print('factors:', ph_units.DefaultToTHz, v_ops._factor, sep='\n')

print(sq_mod[:,:,0]) # x direction
print()
print(sq_mod[:,:,1]) # y direction
print()
print(sq_mod[:,:,2]) # z direction
print()

gv = phonopy.phonon.group_velocity.GroupVelocity(ph_data.dynamical_matrix, frequency_factor_to_THz=ph_data.unit_conversion_factor)

gv.run(qpoint)
print('group velocity')
print(np.sqrt(np.sum(gv.group_velocities**2, axis=2))[0]*velocity_factor, end='\n\n')

print('velocity operator')
print(np.diag(np.sqrt(np.sum(sq_mod, axis=2))))
