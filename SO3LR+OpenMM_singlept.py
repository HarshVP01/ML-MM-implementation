import os
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu,cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential
from openmmml.models.asepotential import ASEPotentialImplFactory
from ase import Atoms as ASEAtoms
from so3lr import So3lrCalculator

MLPotential.registerImplFactory('ase', ASEPotentialImplFactory())

# =====================
# System setup
# =====================
psf    = app.CharmmPsfFile('/g/data/v15/hp2306/charmm/Peptides_final/pep_PRO_ionized.psf')
params = app.CharmmParameterSet(
    '/g/data/v15/hp2306/charmm/toppar/par_all36m_prot.prm',
    '/g/data/v15/hp2306/charmm/toppar/par_water_ions_namd.prm'
)
pdb = app.PDBFile('/g/data/v15/hp2306/charmm/Peptides_final/pep_PRO_ionized.pdb')

if pdb.topology.getPeriodicBoxVectors() is not None:
    box_vecs = pdb.topology.getPeriodicBoxVectors()
    psf.setBox(
        box_vecs[0][0].value_in_unit(unit.nanometers),
        box_vecs[1][1].value_in_unit(unit.nanometers),
        box_vecs[2][2].value_in_unit(unit.nanometers),
    )
#else:
#    psf.setBox(3.9486, 4.3773, 3.2252)         # place holder for box dimensions

topology  = psf.topology
positions = pdb.positions

MM_RESIDUE_NAMES = {'TIP3', 'TIP3W', 'HOH', 'WAT', 'SOD', 'CLA',
                    'POT', 'NA', 'CL', 'K', 'MG', 'CAL', 'ZN'}

ml_atoms = []
mm_atoms = []
for atom in topology.atoms():
    resname = atom.residue.name.strip().upper()
    if resname in MM_RESIDUE_NAMES:
        mm_atoms.append(atom.index)
    else:
        ml_atoms.append(atom.index)

# =====================
# MM system
# =====================
mm_system = psf.createSystem(
    params,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.2 * unit.nanometers,
    switchDistance=1.0 * unit.nanometers,
    constraints=None,
    ewaldErrorTolerance=1e-6,
)

namd_alpha = 0.257952 * 10
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.NonbondedForce):
        f.setPMEParameters(namd_alpha, 40, 45, 36)
        f.setUseDispersionCorrection(False)

# =====================
# SO3LR calculator
# =====================
ml_element_symbols = [
    atom.element.symbol
    for atom in topology.atoms()
    if atom.index in set(ml_atoms)
]

ase_atoms = ASEAtoms(
    symbols=ml_element_symbols,
    positions=[[0.0, 0.0, 0.0]] * len(ml_atoms),
    pbc=False,
)
ase_atoms.calc = So3lrCalculator(
    lr_cutoff=12.,
    dispersion_energy_cutoff_lr_damping=2.,
    calculate_stress=False,
    calculate_hessian=False,
    dtype=np.float32,
)

# =====================
# Mixed ML/MM system
# =====================
ml_system = MLPotential('ase').createMixedSystem(
    topology, mm_system, ml_atoms,
    interpolate=False,
    aseAtoms=ase_atoms,
)

# =====================
# Single point calculation
# =====================
platform = openmm.Platform.getPlatformByName('CPU')
integrator = openmm.VerletIntegrator(0.001)
simulation = app.Simulation(topology, ml_system, integrator, platform,
                             {'Threads': '12'})
simulation.context.setPositions(positions)
state        = simulation.context.getState(getEnergy=True, getForces=True)
total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

# =====================
# Energy decomposition
# =====================
for i in range(ml_system.getNumForces()):
    ml_system.getForce(i).setForceGroup(i)

integrator2 = openmm.VerletIntegrator(0.001)
simulation2  = app.Simulation(topology, ml_system, integrator2, platform,
                               {'Threads': '12'})
simulation2.context.setPositions(positions)

bond_kj = angle_kj = dihed_kj = cmap_kj = nonbond_kj = so3lr_kj = 0.0
for i in range(ml_system.getNumForces()):
    st         = simulation2.context.getState(getEnergy=True, groups={i})
    energy_kj  = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    force_name = type(ml_system.getForce(i)).__name__
    if   force_name == 'HarmonicBondForce':                         bond_kj    += energy_kj
    elif force_name == 'HarmonicAngleForce':                        angle_kj   += energy_kj
    elif force_name in ('PeriodicTorsionForce', 'CustomTorsionForce'): dihed_kj += energy_kj
    elif force_name == 'CMAPTorsionForce':                          cmap_kj    += energy_kj
    elif force_name == 'NonbondedForce':                            nonbond_kj += energy_kj
    elif force_name == 'PythonForce':                               so3lr_kj   += energy_kj

# =====================
# Results
# =====================
print(f"\n=== SO3LR+OpenMM Single Point Results ===")
print(f"{'Term':<25} {'kJ/mol':>16} {'kcal/mol':>16}")
print("-" * 60)
print(f"{'BOND':<25} {bond_kj:>16.4f} {bond_kj/4.184:>16.4f}")
print(f"{'ANGLE':<25} {angle_kj:>16.4f} {angle_kj/4.184:>16.4f}")
print(f"{'DIHED+IMPRP':<25} {dihed_kj:>16.4f} {dihed_kj/4.184:>16.4f}")
print(f"{'CMAP':<25} {cmap_kj:>16.4f} {cmap_kj/4.184:>16.4f}")
print(f"{'ELECT+VDW':<25} {nonbond_kj:>16.4f} {nonbond_kj/4.184:>16.4f}")
print(f"{'SO3LR (peptide)':<25} {so3lr_kj:>16.4f} {so3lr_kj/4.184:>16.4f}")
print("-" * 60)
print(f"{'TOTAL':<25} {total_energy:>16.4f} {total_energy/4.184:>16.4f}")

forces = state.getForces(asNumpy=True).value_in_unit(
    unit.kilojoules_per_mole / unit.nanometer)
print(f"\n=== Forces ===")
print(f"{'Atom':>6} {'Name':>6} {'Region':>6} "
      f"{'Fx':>14} {'Fy':>14} {'Fz':>14} {'|F|':>14}")
print("-" * 74)
for atom in topology.atoms():
    i = atom.index
    fx, fy, fz = forces[i]
    print(f"{i:>6} {atom.name:>6} {'ML' if i in set(ml_atoms) else 'MM':>6} "
          f"{fx:>14.4f} {fy:>14.4f} {fz:>14.4f} "
          f"{np.sqrt(fx**2+fy**2+fz**2):>14.4f}")
