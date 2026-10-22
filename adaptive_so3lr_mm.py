#!/usr/bin/env python
"""
Adaptive SO3LR/MM molecular dynamics of a solvated peptide.

The peptide and nearby water molecules are treated with SO3LR; the rest of
the system with CHARMM36m/TIP3P. Waters within R_OFF of the peptide form the
ML region, with a switched buffer between R_ON and R_OFF.

Example:
  python adaptive_so3lr_mm.py --name GTG --psf GTG.psf --pdb GTG.pdb \\
      --prm par_all36m_prot.prm par_water_ions_namd.prm --replica 1
"""

import os
import sys
import time
import socket
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Adaptive SO3LR/MM molecular dynamics',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group('input / output')
    g.add_argument('--name', default='peptide', help='system name (output prefix)')
    g.add_argument('--psf', required=True, help='CHARMM PSF file')
    g.add_argument('--pdb', required=True, help='PDB file with box (CRYST1)')
    g.add_argument('--prm', nargs='+', required=True, help='CHARMM parameter files')
    g.add_argument('--replica', type=int, default=1)
    g.add_argument('--seed', type=int, default=None, help='random seed (default: 1000 * replica)')
    g.add_argument('--out', default=None, help='output directory (default: ./<name>_rep<replica>)')

    g = p.add_argument_group('adaptive region')
    g.add_argument('--r-on', type=float, default=3.5, help='Angstrom')
    g.add_argument('--r-off', type=float, default=6.0, help='Angstrom')
    g.add_argument('--n-shell-max', type=int, default=150, help='maximum number of ML waters')
    g.add_argument('--cancellation', choices=['subtract', 'remove'], default='subtract')
    g.add_argument('--lr-cutoff', type=float, default=100.0, help='SO3LR long-range cutoff (Angstrom)')

    g = p.add_argument_group('MD protocol')
    g.add_argument('--temperature', type=float, default=300.0, help='K')
    g.add_argument('--pressure', type=float, default=1.0, help='bar')
    g.add_argument('--barostat-freq', type=int, default=25)
    g.add_argument('--mm-npt-steps', type=int, default=100_000)
    g.add_argument('--mm-npt-dt', type=float, default=2.0, help='fs')
    g.add_argument('--warmup-steps', type=int, default=10_000)
    g.add_argument('--warmup-dt', type=float, default=0.1, help='fs')
    g.add_argument('--ramp-dts', type=float, nargs='+', default=[0.3, 0.5, 1.0], help='fs')
    g.add_argument('--ramp-max-steps', type=int, default=20_000)
    g.add_argument('--ramp-consec', type=int, default=500)
    g.add_argument('--warmup-fmax', type=float, default=10_000.0, help='kJ/mol/nm')
    g.add_argument('--dt', type=float, default=1.0, help='fs')
    g.add_argument('--friction', type=float, default=10.0, help='1/ps')
    g.add_argument('--nvt-steps', type=int, default=500_000)
    g.add_argument('--npt-steps', type=int, default=500_000)
    g.add_argument('--prod-steps', type=int, default=15_000_000)
    g.add_argument('--report-every', type=int, default=10_000)
    g.add_argument('--save-every', type=int, default=1_000)
    g.add_argument('--checkpoint-every', type=int, default=500_000)

    g = p.add_argument_group('hardware')
    g.add_argument('--platform', default='CUDA', choices=['CUDA', 'OpenCL', 'CPU'])
    g.add_argument('--cuda-device', default='0')
    g.add_argument('--jax-cache', default=None, help='JAX compilation cache directory')
    return p.parse_args()


args = parse_args()
if args.seed is None:
    args.seed = 1000 * args.replica
if args.out is None:
    args.out = f'{args.name}_rep{args.replica}'
os.makedirs(args.out, exist_ok=True)
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
sys.stdout.reconfigure(write_through=True)

import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
if args.jax_cache:
    try:
        jax.config.update('jax_compilation_cache_dir', args.jax_cache)
    except Exception:
        from jax.experimental.compilation_cache import compilation_cache
        compilation_cache.set_cache_dir(args.jax_cache)
from ase.data import atomic_numbers          # noqa: E402
from jax_md import space                     # noqa: E402
from so3lr import to_jax_md, So3lrPotential  # noqa: E402
import openmm                                # noqa: E402
import openmm.app as app                     # noqa: E402
import openmm.unit as unit                   # noqa: E402

import mm_setup                              # noqa: E402
from mm_setup import pbc_diff, switching, dummy_slot_offsets, make_whole  # noqa: E402
from oniom_inner import InnerMMContext       # noqa: E402

EV_TO_KJMOL = 96.4853321                     # eV -> kJ/mol
EV_ANG_TO_KJMOL_NM = 964.853321              # eV/A -> kJ/mol/nm
KB_KJMOL = 0.00831446261815324               # kJ/mol/K

PREFIX = f'{args.name}_rep{args.replica}'
OUT_DCD_EQUIL = f'{args.out}/{PREFIX}_npt_equil.dcd'
OUT_DCD_PROD = f'{args.out}/{PREFIX}_prod.dcd'
OUT_LOG = f'{args.out}/{PREFIX}_prod.csv'
OUT_CKPT = f'{args.out}/{PREFIX}_prod.chk'


# ---------------------------------------------------------------------------
# SO3LR
# ---------------------------------------------------------------------------
class So3lrEvaluator:
    """SO3LR energy and forces for a fixed set of atoms (free space)."""

    def __init__(self, species, lr_cutoff=100.0):
        displacement, _ = space.free()
        self.neighbor_fn, self.neighbor_fn_lr, energy_fn = to_jax_md(
            potential=So3lrPotential(lr_cutoff=lr_cutoff),
            displacement_or_metric=displacement, box_size=None,
            species=jnp.array(species, dtype=jnp.int32),
            capacity_multiplier=1.25, buffer_size_multiplier_sr=1.25,
            buffer_size_multiplier_lr=1.25, minimum_cell_size_multiplier_sr=1.0,
            disable_cell_list=True, fractional_coordinates=False)

        def energy(R, nb_idx, nblr_idx):
            return energy_fn(R, neighbor=nb_idx, neighbor_lr=nblr_idx, box=None)

        self._value_and_grad = jax.jit(jax.value_and_grad(energy))
        self._update_sr = jax.jit(lambda R, nb: self.neighbor_fn.update(R, nb))
        self._update_lr = jax.jit(lambda R, nb: self.neighbor_fn_lr.update(R, nb))
        self.nbrs = None
        self.nbrs_lr = None

    def evaluate(self, pos_ang):
        """Positions in Angstrom -> (energy kJ/mol, forces kJ/mol/nm)."""
        R = jnp.asarray(pos_ang, dtype=jnp.float32)
        if self.nbrs is None:
            self.nbrs = self.neighbor_fn.allocate(R, box=None)
            self.nbrs_lr = self.neighbor_fn_lr.allocate(R, box=None)
        else:
            self.nbrs = self._update_sr(R, self.nbrs)
            self.nbrs_lr = self._update_lr(R, self.nbrs_lr)
            if self.nbrs.did_buffer_overflow or self.nbrs_lr.did_buffer_overflow:
                self.nbrs = self.neighbor_fn.allocate(R, box=None)
                self.nbrs_lr = self.neighbor_fn_lr.allocate(R, box=None)
        e, grad = self._value_and_grad(R, self.nbrs.idx, self.nbrs_lr.idx)
        return float(e) * EV_TO_KJMOL, -np.asarray(grad, dtype=np.float64) * EV_ANG_TO_KJMOL_NM


class So3lrPool:
    """SO3LR on the peptide + ML waters; one evaluator per number of waters."""

    def __init__(self, pept_species, lr_cutoff):
        self.pept_species = np.asarray(pept_species)
        self.lr_cutoff = lr_cutoff
        self._ev = {}

    def evaluate(self, pept_pos, water_pos):
        n_w = len(water_pos)
        if n_w not in self._ev:
            species = np.concatenate([self.pept_species, np.tile([8, 1, 1], n_w)])
            self._ev[n_w] = So3lrEvaluator(species, self.lr_cutoff)
        R = np.concatenate([pept_pos, np.asarray(water_pos).reshape(-1, 3)]).astype(np.float32)
        e, F = self._ev[n_w].evaluate(R)
        n_p = len(self.pept_species)
        return e, F[:n_p], F[n_p:].reshape(n_w, 3, 3)


# ---------------------------------------------------------------------------
# ML region
# ---------------------------------------------------------------------------
class ShellManager:
    """Selects the ML waters and their switching weights."""

    def __init__(self, pept_heavy, water_mols, r_on, r_off, n_slots):
        self.pept_heavy = np.asarray(pept_heavy)
        wm = np.asarray(water_mols)
        self.w_O, self.w_H1, self.w_H2 = wm[:, 0], wm[:, 1], wm[:, 2]
        self.r_on, self.r_off = r_on, r_off
        self.n_slots = n_slots
        self.slot_mol = np.full(n_slots, -1)
        self.mol_slot = {}
        self.max_shell = 0

    def update(self, pos, box):
        """pos, box in Angstrom -> (weights, water positions, n_inner, n_buffer)."""
        heavy = pos[self.pept_heavy]
        com = heavy.mean(axis=0)
        reach = np.sqrt(((heavy - com) ** 2).sum(axis=1)).max()

        O = pos[self.w_O]
        d_com = np.sqrt((pbc_diff(O - com, box) ** 2).sum(axis=1))
        cand = np.nonzero(d_com < self.r_off + reach + 1.0)[0]

        if cand.size:
            diff = pbc_diff(O[cand, None, :] - heavy[None, :, :], box)
            d = np.sqrt((diff ** 2).sum(axis=-1))
            jmin = d.argmin(axis=1)
            dmin = d[np.arange(cand.size), jmin]
            keep = dmin < self.r_off
            mols, dmin, jmin = cand[keep], dmin[keep], jmin[keep]
        else:
            mols, dmin, jmin = np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)

        self.max_shell = max(self.max_shell, mols.size)
        if mols.size > self.n_slots:
            raise RuntimeError(f'{mols.size} waters within R_OFF; increase --n-shell-max')

        new, old = set(mols.tolist()), set(self.mol_slot)
        if new != old:
            for m in old - new:
                self.slot_mol[self.mol_slot.pop(m)] = -1
            free = np.nonzero(self.slot_mol < 0)[0]
            for m, s in zip(sorted(new - old), free):
                self.slot_mol[s] = m
                self.mol_slot[m] = s

        sw = np.zeros(self.n_slots)
        slot_pos = np.zeros((self.n_slots, 3, 3))
        if mols.size:
            slots = np.array([self.mol_slot[m] for m in mols])
            near = heavy[jmin]
            Ow = near + pbc_diff(pos[self.w_O[mols]] - near, box)
            slot_pos[slots, 0] = Ow
            slot_pos[slots, 1] = Ow + pbc_diff(pos[self.w_H1[mols]] - Ow, box)
            slot_pos[slots, 2] = Ow + pbc_diff(pos[self.w_H2[mols]] - Ow, box)
            sw[slots] = switching(dmin, self.r_on, self.r_off)
        n_inner = int(np.sum(dmin <= self.r_on))
        return sw, slot_pos, n_inner, int(mols.size) - n_inner


def clip_rows(F, fmax):
    """Limit the norm of each row of F to fmax (no limit if fmax is 0)."""
    if not fmax:
        return F
    n = np.sqrt((F ** 2).sum(axis=-1))
    mask = n > fmax
    if np.any(mask):
        F = F.copy()
        F[mask] *= (fmax / n[mask])[..., None]
    return F


class ForceCoupler:
    """Adds the ML/MM correction to the OpenMM system as external forces."""

    def __init__(self, system, pept_atoms, water_mols):
        self.pept_atoms = np.asarray(pept_atoms)
        self.water_mols = water_mols

        self.pept_force = self._new_force()
        for idx in self.pept_atoms:
            self.pept_force.addParticle(int(idx), [0.0, 0.0, 0.0])
        self.pept_force.setForceGroup(31)
        system.addForce(self.pept_force)

        self.water_force = self._new_force()
        self.water_pidx = {}
        for mol in water_mols:
            for atom in mol:
                self.water_pidx[atom] = self.water_force.addParticle(atom, [0.0, 0.0, 0.0])
        self.water_force.setForceGroup(31)
        system.addForce(self.water_force)
        self.active = set()

    @staticmethod
    def _new_force():
        f = openmm.CustomExternalForce('-(fx*x+fy*y+fz*z)')
        for name in ('fx', 'fy', 'fz'):
            f.addPerParticleParameter(name)
        return f

    def apply(self, ctx, dF_pept, dF_slots, sw, slot_mol, fmax):
        F_pept = clip_rows(dF_pept, fmax)
        for i, idx in enumerate(self.pept_atoms):
            self.pept_force.setParticleParameters(i, int(idx), F_pept[i].tolist())
        self.pept_force.updateParametersInContext(ctx)

        occ = np.nonzero(slot_mol >= 0)[0]
        F_w = clip_rows(sw[occ, None, None] * dF_slots[occ], fmax)
        new_active = set()
        for j, s in enumerate(occ):
            for k, atom in enumerate(self.water_mols[slot_mol[s]]):
                self.water_force.setParticleParameters(self.water_pidx[atom], atom,
                                                       F_w[j, k].tolist())
                new_active.add(atom)
        for atom in self.active - new_active:
            self.water_force.setParticleParameters(self.water_pidx[atom], atom, [0.0, 0.0, 0.0])
        self.active = new_active
        self.water_force.updateParametersInContext(ctx)


class MMBarostat:
    """Monte Carlo barostat; volume moves are accepted using the MM energy."""

    def __init__(self, context, pressure_bar, temperature_k, seed, groups=frozenset(range(31))):
        mols = context.getMolecules()
        self.atom_mol = np.empty(context.getSystem().getNumParticles(), dtype=int)
        for k, m in enumerate(mols):
            self.atom_mol[list(m)] = k
        self.counts = np.bincount(self.atom_mol).astype(float)
        self.n_mol = len(mols)
        self.p = pressure_bar * 0.0602214076               # bar -> kJ/mol/nm^3
        self.kT = KB_KJMOL * temperature_k
        self.groups = set(groups)
        self.rng = np.random.default_rng(seed)
        bv = context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        self.dv = 0.01 * float(np.prod(np.diag(bv)))
        self.n_try = self.n_acc = 0

    def _energy(self, ctx):
        return ctx.getState(getEnergy=True, groups=self.groups) \
                  .getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def step(self, ctx):
        st = ctx.getState(getPositions=True, getEnergy=True, groups=self.groups)
        e0 = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        v0 = float(np.prod(np.diag(bv)))
        dv = self.dv * (2.0 * self.rng.random() - 1.0)
        v1 = v0 + dv
        f = (v1 / v0) ** (1.0 / 3.0)
        cen = np.stack([np.bincount(self.atom_mol, weights=pos[:, d]) for d in range(3)], 1) \
            / self.counts[:, None]
        ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(f * bv[i])) for i in range(3)])
        ctx.setPositions(pos + (f - 1.0) * cen[self.atom_mol])
        w = self._energy(ctx) - e0 + self.p * dv - self.n_mol * self.kT * np.log(v1 / v0)
        if w > 0.0 and self.rng.random() > np.exp(-w / self.kT):
            ctx.setPeriodicBoxVectors(*[openmm.Vec3(*bv[i]) for i in range(3)])
            ctx.setPositions(pos)
        else:
            self.n_acc += 1
        self.n_try += 1
        if self.n_try >= 10:
            if self.n_acc < 0.25 * self.n_try:
                self.dv /= 1.1
            elif self.n_acc > 0.75 * self.n_try:
                self.dv = min(self.dv * 1.1, 0.3 * v1)
            self.n_try = self.n_acc = 0


class AdaptiveEngine:
    """Computes and applies the adaptive ML/MM forces at each step."""

    def __init__(self, so3lr, inner, shell, coupler, pept_atoms, n_slots, unwrap_tree):
        self.so3lr, self.inner, self.shell, self.coupler = so3lr, inner, shell, coupler
        self.pept_atoms = np.asarray(pept_atoms)
        self.n_slots = n_slots
        self.unwrap_child, self.unwrap_parent = unwrap_tree
        self.dummy_off = dummy_slot_offsets(n_slots)

    def cluster_geometry(self, pos, box):
        pos = make_whole(pos.copy(), box, self.unwrap_child, self.unwrap_parent)
        sw, slot_pos, n_inner, n_buffer = self.shell.update(pos, box)
        pept_pos = pos[self.pept_atoms]
        empty = self.shell.slot_mol < 0
        slot_pos[empty] = (pept_pos.mean(axis=0) + self.dummy_off)[empty]
        return sw, pept_pos, slot_pos, n_inner, n_buffer

    def compute_and_apply(self, ctx, fmax=0.0):
        st = ctx.getState(getPositions=True, enforcePeriodicBox=False)
        bv = st.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
        box = np.array([bv[0][0], bv[1][1], bv[2][2]])
        pos = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))

        sw, pept_pos, slot_pos, n_inner, n_buffer = self.cluster_geometry(pos, box)
        self.inner.set_slots(self.shell.slot_mol)

        occ = np.nonzero(self.shell.slot_mol >= 0)[0]
        e_ml, F_ml_pept, F_ml_w = self.so3lr.evaluate(pept_pos, slot_pos[occ])
        if not (np.all(np.isfinite(F_ml_pept)) and np.all(np.isfinite(F_ml_w))):
            raise FloatingPointError('non-finite SO3LR forces')
        F_ml_slots = np.zeros((self.n_slots, 3, 3))
        F_ml_slots[occ] = F_ml_w

        F_mm_pept, F_mm_slots = self.inner.evaluate(pept_pos, slot_pos)
        self.coupler.apply(ctx, F_ml_pept - F_mm_pept, F_ml_slots - F_mm_slots,
                           sw, self.shell.slot_mol, fmax)
        return dict(e_ml=e_ml, n_inner=n_inner, n_buffer=n_buffer,
                    fmax_ml=float(np.sqrt((F_ml_pept ** 2).sum(axis=1)).max()))


# ---------------------------------------------------------------------------
# Set-up
# ---------------------------------------------------------------------------
for f in [args.psf, args.pdb] + list(args.prm):
    if not os.path.exists(f):
        sys.exit(f'ERROR: file not found: {f}')

psf, params, pdb, box_nm = mm_setup.load_inputs(args.psf, args.pdb, args.prm)
topology = psf.topology
n_atoms = topology.getNumAtoms()
pept_atoms, pept_heavy, water_mols = mm_setup.classify_atoms(topology)
unwrap_tree = mm_setup.peptide_unwrap_tree(
    [(b[0].index, b[1].index) for b in topology.bonds()], pept_atoms)
atoms = list(topology.atoms())
pept_species = np.array([atomic_numbers[atoms[i].element.symbol] for i in pept_atoms])

print('=' * 72)
print(f'Adaptive SO3LR/MM  --  {args.name}  replica {args.replica}  seed {args.seed}')
print(f'R_ON = {args.r_on} A   R_OFF = {args.r_off} A')
print(f'Atoms: {n_atoms}   peptide: {len(pept_atoms)}   waters: {len(water_mols)}')
print(f'Output: {args.out}   node: {socket.gethostname()}')
print('=' * 72)

mm_system = mm_setup.create_mm_system(psf, params)
mm_setup.remove_peptide_internal_mm(mm_system, pept_atoms,
                                    keep_bonded=(args.cancellation == 'subtract'))

platform = openmm.Platform.getPlatformByName(args.platform)
if args.platform == 'CUDA':
    platform_props = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': args.cuda_device}
elif args.platform == 'OpenCL':
    platform_props = {'OpenCLPrecision': 'mixed', 'OpenCLDeviceIndex': args.cuda_device}
else:
    platform_props = {}

inner = InnerMMContext(mm_system, pept_atoms, water_mols, args.n_shell_max,
                       platform, platform_props)
coupler = ForceCoupler(mm_system, pept_atoms, water_mols)
shell = ShellManager(pept_heavy, water_mols, args.r_on, args.r_off, args.n_shell_max)

n_cm = sum(isinstance(f, openmm.CMMotionRemover) for f in mm_system.getForces())
n_dof = 3 * n_atoms - mm_system.getNumConstraints() - 3 * n_cm

pos0_ang = np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
box0_ang = box_nm * 10.0

so3lr = So3lrPool(pept_species, args.lr_cutoff)
engine = AdaptiveEngine(so3lr, inner, shell, coupler, pept_atoms, args.n_shell_max, unwrap_tree)


# ---------------------------------------------------------------------------
# MD helpers
# ---------------------------------------------------------------------------
def make_simulation(dt_fs, friction, seed, positions, velocities=True, system=None,
                    box_vectors=None):
    integ = openmm.LangevinMiddleIntegrator(args.temperature * unit.kelvin,
                                            friction / unit.picosecond,
                                            dt_fs * unit.femtoseconds)
    integ.setRandomNumberSeed(seed)
    sim = app.Simulation(topology, mm_system if system is None else system,
                         integ, platform, platform_props)
    if box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(positions)
    if velocities:
        sim.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, seed)
    return sim


def temperature(ctx):
    ke = ctx.getState(getEnergy=True).getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    return 2.0 * ke / (n_dof * KB_KJMOL)


def run_adaptive(sim, n_steps, dt_fs, label, fmax=0.0, report_every=None, log_fh=None,
                 gate=None, barostat=None):
    """Run n_steps of adaptive SO3LR/MM. With gate, stop once converged."""
    if report_every:
        print(f'\n  {label}')
        print(f'  {"step":>10} {"t (ps)":>9} {"T (K)":>7} {"E_ML (kJ/mol)":>14} '
              f'{"inner":>6} {"buffer":>6} {"ms/step":>8}')
    consec = 0
    t_blk = time.perf_counter()
    for step in range(1, n_steps + 1):
        info = engine.compute_and_apply(sim.context, fmax)
        sim.step(1)
        if barostat is not None and step % args.barostat_freq == 0:
            barostat.step(sim.context)

        if gate is not None:
            consec = consec + 1 if info['fmax_ml'] < gate['thresh'] else 0
            if consec >= gate['consec']:
                return step

        if report_every and step % report_every == 0:
            now = time.perf_counter()
            ms = (now - t_blk) / report_every * 1000.0
            t_blk = now
            T = temperature(sim.context)
            t_ps = step * dt_fs / 1000.0
            print(f'  {step:>10} {t_ps:>9.1f} {T:>7.1f} {info["e_ml"]:>14.2f} '
                  f'{info["n_inner"]:>6} {info["n_buffer"]:>6} {ms:>8.1f}')
            if log_fh:
                log_fh.write(f'{step},{t_ps:.3f},{T:.3f},{info["e_ml"]:.4f},'
                             f'{info["n_inner"]},{info["n_buffer"]},{ms:.3f}\n')
                log_fh.flush()
    return None if gate is not None else n_steps


def peptide_restraint(positions_nm, k=1.0e4):
    """Harmonic positional restraint on peptide heavy atoms."""
    r = openmm.CustomExternalForce('k_rest*periodicdistance(x, y, z, x0, y0, z0)^2')
    r.addGlobalParameter('k_rest', k)
    for name in ('x0', 'y0', 'z0'):
        r.addPerParticleParameter(name)
    for idx in pept_heavy:
        r.addParticle(int(idx), positions_nm[idx].tolist())
    return r


def peptide_rmsd_restraint(positions_nm, k=1.0e4):
    """Restraint on the peptide heavy-atom RMSD (used with the barostat)."""
    rmsd = openmm.RMSDForce(positions_nm, [int(i) for i in pept_heavy])
    cv = openmm.CustomCVForce('k_rmsd*rmsd^2')
    cv.addGlobalParameter('k_rmsd', k * len(pept_heavy))
    cv.addCollectiveVariable('rmsd', rmsd)
    return cv


# ---------------------------------------------------------------------------
# STEP 0: SO3LR relaxation of the peptide
# ---------------------------------------------------------------------------
print('\nSTEP 0: SO3LR relaxation of the peptide')
vac = So3lrEvaluator(pept_species, args.lr_cutoff)
x = pos0_ang[pept_atoms].copy()
for it in range(1, 10_001):
    e, F = vac.evaluate(x)
    fmax = np.abs(F).max()
    if fmax < 50.0:
        break
    x += np.clip(F, -5000.0, 5000.0) * 1.0e-5
print(f'  E = {e:.3f} kJ/mol after {it} iterations')
del vac

pos_nm = pos0_ang / 10.0
pos_nm[pept_atoms] = x / 10.0

# ---------------------------------------------------------------------------
# STEP 1: MM minimisation
# ---------------------------------------------------------------------------
print('\nSTEP 1: MM minimisation')
prep_system = mm_setup.create_mm_system(psf, params)
prep_system.addForce(peptide_restraint(pos_nm))
sim = make_simulation(1.0, 1.0, args.seed, pos_nm, velocities=False, system=prep_system)
sim.minimizeEnergy(tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=0)
st = sim.context.getState(getPositions=True, getEnergy=True)
print(f'  E = {st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole):.2f} kJ/mol')
positions = st.getPositions()
del sim

# ---------------------------------------------------------------------------
# STEP 1b: MM solvent relaxation
# ---------------------------------------------------------------------------
print('\nSTEP 1b: MM solvent relaxation')
sim = make_simulation(0.1, 10.0, args.seed + 1, positions, system=prep_system)
for blk in range(1, 21):
    sim.step(5000)
    p = np.asarray(sim.context.getState(getPositions=True)
                   .getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    O = p[[w[0] for w in water_mols]]
    d = pbc_diff(O[:, None, :] - p[pept_heavy][None, :, :], box0_ang)
    if np.sqrt((d ** 2).sum(axis=-1)).min() >= 2.4:
        break
positions = sim.context.getState(getPositions=True).getPositions()
del sim, prep_system

# ---------------------------------------------------------------------------
# STEP 1c: MM NPT equilibration
# ---------------------------------------------------------------------------
print(f'\nSTEP 1c: MM NPT equilibration  [{args.mm_npt_steps} x {args.mm_npt_dt} fs]')
npt_system = mm_setup.create_mm_system(psf, params)
npt_system.addForce(peptide_rmsd_restraint(pos_nm))
npt_system.addForce(openmm.MonteCarloBarostat(args.pressure * unit.bar,
                                              args.temperature * unit.kelvin,
                                              args.barostat_freq))
sim = make_simulation(args.mm_npt_dt, 1.0, args.seed + 6, positions, system=npt_system)
sim.step(args.mm_npt_steps)
st = sim.context.getState(getPositions=True)
positions, box = st.getPositions(), st.getPeriodicBoxVectors()
mm_system.setDefaultPeriodicBoxVectors(*box)
del sim, npt_system

# ---------------------------------------------------------------------------
# STEP 2: NVT warm-up
# ---------------------------------------------------------------------------
print(f'\nSTEP 2: NVT warm-up  [{args.warmup_steps} x {args.warmup_dt} fs]  '
      f'{time.strftime("%H:%M:%S")}')
sim = make_simulation(args.warmup_dt, args.friction, args.seed + 2, positions)
run_adaptive(sim, args.warmup_steps, args.warmup_dt, 'NVT warm-up', fmax=args.warmup_fmax,
             report_every=max(1, min(500, args.warmup_steps // 5)))
positions = sim.context.getState(getPositions=True).getPositions()
del sim

# ---------------------------------------------------------------------------
# STEP 2b: time-step ramp
# ---------------------------------------------------------------------------
print(f'\nSTEP 2b: time-step ramp  {time.strftime("%H:%M:%S")}')
for i, dt in enumerate(args.ramp_dts):
    sim = make_simulation(dt, args.friction, args.seed + 10 + i, positions)
    n = run_adaptive(sim, args.ramp_max_steps, dt, f'dt = {dt} fs', fmax=args.warmup_fmax,
                     gate=dict(thresh=args.warmup_fmax, consec=args.ramp_consec))
    print(f'  dt = {dt} fs: {n if n else args.ramp_max_steps} steps')
    positions = sim.context.getState(getPositions=True).getPositions()
    del sim

# ---------------------------------------------------------------------------
# STEP 3: NVT equilibration
# ---------------------------------------------------------------------------
print(f'\nSTEP 3: NVT equilibration  [{args.nvt_steps} x {args.dt} fs]  '
      f'{time.strftime("%H:%M:%S")}')
sim = make_simulation(args.dt, args.friction, args.seed + 3, positions)
run_adaptive(sim, args.nvt_steps, args.dt, 'NVT equilibration',
             report_every=max(1, args.nvt_steps // 10))
st = sim.context.getState(getPositions=True, getVelocities=True)
positions, velocities = st.getPositions(), st.getVelocities()
del sim

# ---------------------------------------------------------------------------
# STEP 4: NPT equilibration
# ---------------------------------------------------------------------------
print(f'\nSTEP 4: NPT equilibration  [{args.npt_steps} x {args.dt} fs]  '
      f'{time.strftime("%H:%M:%S")}')
sim = make_simulation(args.dt, args.friction, args.seed + 4, positions, velocities=False)
sim.context.setVelocities(velocities)
sim.reporters.append(app.DCDReporter(OUT_DCD_EQUIL, args.save_every * 10))
baro = MMBarostat(sim.context, args.pressure, args.temperature, args.seed + 4)
run_adaptive(sim, args.npt_steps, args.dt, 'NPT equilibration',
             report_every=max(1, args.npt_steps // 10), barostat=baro)
st = sim.context.getState(getPositions=True, getVelocities=True)
positions, velocities, box = st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors()
del sim

# ---------------------------------------------------------------------------
# STEP 5: NPT production
# ---------------------------------------------------------------------------
print(f'\nSTEP 5: NPT production  [{args.prod_steps} x {args.dt} fs = '
      f'{args.prod_steps * args.dt / 1e6:.2f} ns]  {time.strftime("%H:%M:%S")}')
sim = make_simulation(args.dt, args.friction, args.seed + 5, positions, velocities=False,
                      box_vectors=box)
sim.context.setVelocities(velocities)
sim.reporters.append(app.DCDReporter(OUT_DCD_PROD, args.save_every))
sim.reporters.append(app.CheckpointReporter(OUT_CKPT, args.checkpoint_every))
baro = MMBarostat(sim.context, args.pressure, args.temperature, args.seed + 5)

with open(OUT_LOG, 'w') as log_fh:
    log_fh.write('step,time_ps,T_K,E_ML_kJmol,n_inner,n_buffer,ms_per_step\n')
    t_start = time.perf_counter()
    run_adaptive(sim, args.prod_steps, args.dt, 'NPT production',
                 report_every=args.report_every, log_fh=log_fh, barostat=baro)
sim.saveCheckpoint(OUT_CKPT)
ms_mean = (time.perf_counter() - t_start) / args.prod_steps * 1000.0

print(f'\n{"=" * 72}\nCOMPLETE')
print(f'  performance : {ms_mean:.1f} ms/step ({86400.0 / ms_mean * args.dt / 1e3:.2f} ns/day)')
print(f'  trajectory  : {OUT_DCD_PROD}')
print('=' * 72)
