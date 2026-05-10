# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.stdout.reconfigure(write_through=True)

from ase import Atoms as ASEAtoms
from jax_md import space, quantity
from so3lr import to_jax_md, So3lrPotential
import openmm
import openmm.app as app
import openmm.unit as unit

# ============================================================
# USER CONFIGURATION
# ============================================================
BASE     = '/g/data/v15/hp2306/NNP_MM/test_polyala/charmm-gui/charmm-gui-7683686462'
PSF_FILE = f'{BASE}/step3_pbcsetup.psf'
PDB_FILE = f'{BASE}/step3_pbcsetup.pdb'
PRM1     = '/g/data/v15/hp2306/charmm/toppar/par_all36m_prot.prm'
PRM2     = '/g/data/v15/hp2306/charmm/toppar/par_water_ions_namd.prm'
OUT      = '/scratch/v15/hp2306'
OUTPUT_FILE = f'{OUT}/mlmm_singlept.txt'

BOX_SIZE   = 4.0    # nm
NB_CUTOFF  = 1.2    # nm
SWITCH_DIST = 1.0   # nm
NAMD_ALPHA  = 0.257952 * 10
PME_GRID    = 48

MM_RESIDUE_NAMES = {'TIP3','TIP3W','HOH','WAT','SOD','CLA',
                    'POT','NA','CL','K','MG','CAL','ZN'}

# ============================================================
# Load topology
# ============================================================
psf      = app.CharmmPsfFile(PSF_FILE)
params   = app.CharmmParameterSet(PRM1, PRM2)
pdb      = app.PDBFile(PDB_FILE)
psf.setBox(BOX_SIZE, BOX_SIZE, BOX_SIZE)
topology  = psf.topology
positions = pdb.positions
n_atoms   = topology.getNumAtoms()

ml_atoms = [a.index for a in topology.atoms()
            if a.residue.name.strip().upper() not in MM_RESIDUE_NAMES]
ml_set   = set(ml_atoms)
ml_arr   = np.array(ml_atoms)
n_ml     = len(ml_atoms)
ml_syms  = [a.element.symbol for a in topology.atoms()
            if a.index in ml_set]
ml_nums  = np.array([ASEAtoms([s]).get_atomic_numbers()[0]
                     for s in ml_syms])

pos_ang_all = np.array([[p.x, p.y, p.z]
    for p in positions.value_in_unit(unit.angstrom)])
ml_pos = pos_ang_all[ml_arr]

print(f"System: {n_atoms} atoms  |  ML: {n_ml}  |  MM: {n_atoms-n_ml}")

# ============================================================
# SO3LR initialisation
# ============================================================
box = None
displacement, shift = space.free()

neighbor_fn, neighbor_fn_lr, energy_fn = to_jax_md(
    potential=So3lrPotential(),
    displacement_or_metric=displacement,
    box_size=box,
    species=jnp.array(ml_nums, dtype=jnp.int32),
    capacity_multiplier=1.25,
    buffer_size_multiplier_sr=1.25,
    buffer_size_multiplier_lr=1.25,
    minimum_cell_size_multiplier_sr=1.0,
    disable_cell_list=True,
    fractional_coordinates=False,
)

energy_fn_jit = jax.jit(energy_fn)
force_fn_jit  = jax.jit(quantity.force(energy_fn))

pos_jax = jnp.array(ml_pos, dtype=jnp.float32)
nbrs    = neighbor_fn.allocate(pos_jax, box=box)
nbrs_lr = neighbor_fn_lr.allocate(pos_jax, box=box)

t0 = time.perf_counter()
_ = energy_fn_jit(pos_jax, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
_ = force_fn_jit(pos_jax, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=box)
jax.block_until_ready(_)
t_jit = time.perf_counter() - t0
print(f"JIT ready in {t_jit:.1f} s "
      f"({'cache hit' if t_jit < 30 else 'compiled from scratch'})")

# ============================================================
# SO3LR single point
# ============================================================
t0 = time.perf_counter()
e_ev     = energy_fn_jit(pos_jax, neighbor=nbrs.idx,
                          neighbor_lr=nbrs_lr.idx, box=box)
f_ev_ang = force_fn_jit(pos_jax, neighbor=nbrs.idx,
                         neighbor_lr=nbrs_lr.idx, box=box)
jax.block_until_ready(f_ev_ang)
t_so3lr = time.perf_counter() - t0

e_so3lr = float(e_ev) * 96.485          # eV -> kJ/mol
f_so3lr = np.array(f_ev_ang) * 964.85   # eV/Ang -> kJ/mol/nm

print(f"E_SO3LR = {e_so3lr:.4f} kJ/mol  "
      f"F_max = {np.max(np.abs(f_so3lr)):.2f} kJ/mol/nm  "
      f"({t_so3lr*1000:.2f} ms)")

# ============================================================
# Build ML/MM OpenMM system
# Fix 1: zero ML-ML nonbonded
# Fix 2: zero ML torsions
# Fix 3: CustomExternalForce for SO3LR forces
# ============================================================
mm_system = psf.createSystem(params,
    nonbondedMethod=app.PME,
    nonbondedCutoff=NB_CUTOFF * unit.nanometers,
    switchDistance=SWITCH_DIST * unit.nanometers,
    constraints=app.AllBonds,
    ewaldErrorTolerance=1e-6)

for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.NonbondedForce):
        f.setPMEParameters(NAMD_ALPHA, PME_GRID, PME_GRID, PME_GRID)
        f.setUseSwitchingFunction(True)
        f.setSwitchingDistance(SWITCH_DIST * unit.nanometer)
        f.setUseDispersionCorrection(False)

# Fix 2: zero ML torsions
n_t = 0
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.PeriodicTorsionForce):
        for j in range(f.getNumTorsions()):
            p1,p2,p3,p4,per,phase,k = f.getTorsionParameters(j)
            if any(p in ml_set for p in [p1,p2,p3,p4]):
                f.setTorsionParameters(j,p1,p2,p3,p4,per,phase,
                    0.0*unit.kilojoules_per_mole)
                n_t += 1

# Fix 1: zero ML-ML nonbonded
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.NonbondedForce):
        existing = {}
        for k in range(f.getNumExceptions()):
            p1,p2,cp,sigma,eps = f.getExceptionParameters(k)
            if p1 in ml_set and p2 in ml_set:
                existing[(min(p1,p2),max(p1,p2))] = k
        n_zer = n_add = 0
        for pair,exc_idx in existing.items():
            p1,p2,cp,sigma,eps = f.getExceptionParameters(exc_idx)
            f.setExceptionParameters(exc_idx,p1,p2,0.0,sigma,0.0)
            n_zer += 1
        ml_list = list(ml_set)
        for a in range(len(ml_list)):
            for b in range(a+1,len(ml_list)):
                pair = (min(ml_list[a],ml_list[b]),
                        max(ml_list[a],ml_list[b]))
                if pair not in existing:
                    f.addException(ml_list[a],ml_list[b],0.0,1.0,0.0)
                    n_add += 1

print(f"Fix 1: zeroed={n_zer} added={n_add} ML-ML NB  "
      f"Fix 2: zeroed {n_t} ML torsions")

# Fix 3: inject SO3LR forces via CustomExternalForce
so3lr_force = openmm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
so3lr_force.addPerParticleParameter("fx")
so3lr_force.addPerParticleParameter("fy")
so3lr_force.addPerParticleParameter("fz")
for idx in ml_atoms:
    so3lr_force.addParticle(idx, [0.0, 0.0, 0.0])
mm_system.addForce(so3lr_force)

# ============================================================
# Inject SO3LR forces and get OpenMM state
# ============================================================
try:
    platform = openmm.Platform.getPlatformByName('CUDA')
    props    = {'CudaPrecision': 'double', 'CudaDeviceIndex': '0'}
except:
    platform = openmm.Platform.getPlatformByName('Reference')
    props    = {}

integ = openmm.VerletIntegrator(1.0*unit.femtoseconds)
sim   = app.Simulation(topology, mm_system, integ, platform, props)
sim.context.setPositions(positions)

_fp = [[0.0,0.0,0.0] for _ in range(n_ml)]
for i in range(n_ml):
    _fp[i][0] = float(f_so3lr[i,0])
    _fp[i][1] = float(f_so3lr[i,1])
    _fp[i][2] = float(f_so3lr[i,2])
    so3lr_force.setParticleParameters(i, ml_atoms[i], _fp[i])
so3lr_force.updateParametersInContext(sim.context)

t0    = time.perf_counter()
state = sim.context.getState(getEnergy=True, getForces=True,
                              getPositions=True)
t_mm  = time.perf_counter() - t0

pos_nm = np.array(state.getPositions(asNumpy=True))
e_omm  = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
f_omm  = np.array(state.getForces(asNumpy=True).value_in_unit(
    unit.kilojoules_per_mole/unit.nanometer))

# Fix 3: E_true = E_omm - E_custom + E_SO3LR
# E_custom = -(f·r) stored by OpenMM
# e_fake   = +f·r  (our computation)
# E_MM     = e_omm + e_fake  (removes CustomExternalForce contribution)
e_fake = float(np.sum(f_so3lr * pos_nm[ml_arr]))
e_mm   = e_omm + e_fake
e_pot  = e_mm + e_so3lr

print(f"E_omm (with fake) = {e_omm:.4f}  "
      f"E_fake = {e_fake:.4f}  "
      f"E_MM = {e_mm:.4f} kJ/mol  ({t_mm*1000:.1f} ms)")

# ============================================================
# Force group decomposition to isolate ML-MM coupling
# ============================================================
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.HarmonicBondForce):       f.setForceGroup(1)
    elif isinstance(f, openmm.HarmonicAngleForce):    f.setForceGroup(2)
    elif isinstance(f, openmm.PeriodicTorsionForce):  f.setForceGroup(3)
    elif isinstance(f, openmm.NonbondedForce):        f.setForceGroup(4)
    elif isinstance(f, openmm.CustomExternalForce):   f.setForceGroup(5)
    else:                                              f.setForceGroup(6)

integ2 = openmm.VerletIntegrator(1.0*unit.femtoseconds)
sim2   = app.Simulation(topology, mm_system, integ2, platform, props)
sim2.context.setPositions(positions)
for i in range(n_ml):
    so3lr_force.setParticleParameters(i, ml_atoms[i], _fp[i])
so3lr_force.updateParametersInContext(sim2.context)

def get_group_energy(ctx, group):
    s = ctx.getState(getEnergy=True, groups={group})
    return s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

e_bonds    = get_group_energy(sim2.context, 1)
e_angles   = get_group_energy(sim2.context, 2)
e_torsions = get_group_energy(sim2.context, 3)
e_nb       = get_group_energy(sim2.context, 4)
e_cust     = get_group_energy(sim2.context, 5)

# Temporarily zero ML nonbonded to get MM-MM only
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.NonbondedForce):
        ml_params_saved = []
        for idx in ml_atoms:
            cp,sigma,eps = f.getParticleParameters(idx)
            ml_params_saved.append((cp,sigma,eps))
            f.setParticleParameters(idx,
                0.0*unit.elementary_charge, sigma,
                0.0*unit.kilojoules_per_mole)
        ml_exc_saved = []
        for k in range(f.getNumExceptions()):
            p1,p2,cp,sig,eps = f.getExceptionParameters(k)
            if p1 in ml_set or p2 in ml_set:
                ml_exc_saved.append((k,p1,p2,cp,sig,eps))
                f.setExceptionParameters(k,p1,p2,
                    0.0*unit.elementary_charge**2,
                    sig, 0.0*unit.kilojoules_per_mole)
        f.updateParametersInContext(sim2.context)

e_nb_mm_only = get_group_energy(sim2.context, 4)

# Restore ML parameters
for i in range(mm_system.getNumForces()):
    f = mm_system.getForce(i)
    if isinstance(f, openmm.NonbondedForce):
        for idx,(cp,sigma,eps) in zip(ml_atoms, ml_params_saved):
            f.setParticleParameters(idx, cp, sigma, eps)
        for k,p1,p2,cp,sig,eps in ml_exc_saved:
            f.setExceptionParameters(k,p1,p2,cp,sig,eps)
        f.updateParametersInContext(sim2.context)

e_ml_mm = e_nb - e_nb_mm_only

# ============================================================
# Results
# ============================================================
print(f"\n{'='*65}")
print(f"RESULTS")
print(f"{'='*65}")
print(f"\nEnergy decomposition (kJ/mol):")
print(f"  {'E_SO3LR (ML intramol)':<35} : {e_so3lr:>14.4f}")
print(f"  {'E_ML-MM (ELECT+VDW coupling)':<35} : {e_ml_mm:>14.4f}")
print(f"  {'E_MM (bonds+angles+NB, MM only)':<35} : {e_mm - e_ml_mm:>14.4f}")
print(f"  {'E_MM+coupling (OpenMM contribution)':<35} : {e_mm:>14.4f}")
print(f"  {'E_pot (true total)':<35} : {e_pot:>14.4f}")

print(f"\nForce group breakdown (kJ/mol):")
print(f"  {'Bonds (ML+MM)':<30} : {e_bonds:>12.4f}")
print(f"  {'Angles (ML+MM)':<30} : {e_angles:>12.4f}")
print(f"  {'Torsions (ML zeroed)':<30} : {e_torsions:>12.4f}")
print(f"  {'NB total (ML-MM + MM-MM)':<30} : {e_nb:>12.4f}")
print(f"  {'NB MM-MM only':<30} : {e_nb_mm_only:>12.4f}")
print(f"  {'ML-MM coupling':<30} : {e_ml_mm:>12.4f}")
print(f"  {'CustomExternalForce (fake)':<30} : {e_cust:>12.4f}")
print(f"  {'Sum (= E_omm check)':<30} : "
      f"{e_bonds+e_angles+e_torsions+e_nb+e_cust:>12.4f}")
print(f"  {'E_omm reference':<30} : {e_omm:>12.4f}")

print(f"\nML atom forces (SO3LR, kJ/mol/nm):")
print(f"  {'Idx':>5} {'El':>4} {'Fx':>12} {'Fy':>12} {'Fz':>12} {'|F|':>12}")
print(f"  {'-'*55}")
f_ml_mag = np.sqrt(np.sum(f_so3lr**2, axis=1))
for i in range(n_ml):
    print(f"  {ml_atoms[i]:>5} {ml_syms[i]:>4} "
          f"{f_so3lr[i,0]:>12.4f} {f_so3lr[i,1]:>12.4f} "
          f"{f_so3lr[i,2]:>12.4f} {f_ml_mag[i]:>12.4f}")

f_mm_all = np.array([f_omm[a.index] for a in topology.atoms()
                     if a.index not in ml_set])
f_mm_mag = np.sqrt(np.sum(f_mm_all**2, axis=1))
print(f"\nMM atom forces summary (CHARMM + ML-MM coupling, kJ/mol/nm):")
print(f"  N atoms : {len(f_mm_all)}")
print(f"  F_max   : {np.max(f_mm_mag):.4f}")
print(f"  F_mean  : {np.mean(f_mm_mag):.4f}")
print(f"  F_rms   : {np.sqrt(np.mean(f_mm_mag**2)):.4f}")

f_total = np.sum(f_so3lr, axis=0) + np.sum(f_mm_all, axis=0)
print(f"\nForce sum (non-zero expected with PME/PBC, kJ/mol/nm):")
print(f"  Sum F = ({f_total[0]:.4f}, {f_total[1]:.4f}, {f_total[2]:.4f})")

print(f"\nTiming:")
print(f"  JIT    : {t_jit:.1f} s")
print(f"  SO3LR  : {t_so3lr*1000:.2f} ms")
print(f"  OpenMM : {t_mm*1000:.2f} ms")

# ============================================================
# Save results
# ============================================================
with open(OUTPUT_FILE, 'w') as fh:
    fh.write("ML/MM Single Point Calculation\n")
    fh.write("=" * 65 + "\n\n")
    fh.write(f"PSF      : {PSF_FILE}\n")
    fh.write(f"PDB      : {PDB_FILE}\n")
    fh.write(f"ML atoms : {n_ml}  MM atoms : {n_atoms-n_ml}\n\n")

    fh.write("Energy decomposition (kJ/mol):\n")
    fh.write(f"  E_SO3LR          = {e_so3lr:.6f}\n")
    fh.write(f"  E_ML-MM coupling = {e_ml_mm:.6f}\n")
    fh.write(f"  E_MM             = {e_mm:.6f}\n")
    fh.write(f"  E_pot            = {e_pot:.6f}\n\n")

    fh.write("Force group breakdown (kJ/mol):\n")
    fh.write(f"  Bonds (ML+MM)    = {e_bonds:.6f}\n")
    fh.write(f"  Angles (ML+MM)   = {e_angles:.6f}\n")
    fh.write(f"  Torsions         = {e_torsions:.6f}\n")
    fh.write(f"  NB total         = {e_nb:.6f}\n")
    fh.write(f"  NB MM-MM only    = {e_nb_mm_only:.6f}\n")
    fh.write(f"  ML-MM coupling   = {e_ml_mm:.6f}\n\n")

    fh.write("ML atom forces (SO3LR, kJ/mol/nm):\n")
    fh.write(f"{'Idx':>6} {'El':>4} {'Fx':>14} {'Fy':>14} "
             f"{'Fz':>14} {'|F|':>14}\n")
    fh.write("-" * 70 + "\n")
    for i in range(n_ml):
        fh.write(f"{ml_atoms[i]:>6} {ml_syms[i]:>4} "
                 f"{f_so3lr[i,0]:>14.6f} {f_so3lr[i,1]:>14.6f} "
                 f"{f_so3lr[i,2]:>14.6f} {f_ml_mag[i]:>14.6f}\n")

    fh.write("\nMM atom forces (CHARMM + ML-MM coupling, kJ/mol/nm):\n")
    fh.write(f"{'Idx':>6} {'El':>4} {'Fx':>14} {'Fy':>14} "
             f"{'Fz':>14} {'|F|':>14}\n")
    fh.write("-" * 70 + "\n")
    for atom in topology.atoms():
        if atom.index not in ml_set:
            fv   = f_omm[atom.index]
            fmag = np.sqrt(np.sum(fv**2))
            fh.write(f"{atom.index:>6} {atom.element.symbol:>4} "
                     f"{fv[0]:>14.6f} {fv[1]:>14.6f} "
                     f"{fv[2]:>14.6f} {fmag:>14.6f}\n")

print(f"\nResults saved to: {OUTPUT_FILE}")
