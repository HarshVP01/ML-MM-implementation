"""MM system set-up and geometry helpers for adaptive SO3LR/MM."""

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

# Residue / atom naming
WATER_RES_NAMES = {'TIP3', 'TIP3W', 'HOH', 'WAT'}
ION_RES_NAMES = {'SOD', 'CLA', 'POT', 'NA', 'CL', 'K', 'MG', 'CAL', 'ZN'}
MM_RESIDUE_NAMES = WATER_RES_NAMES | ION_RES_NAMES

# TIP3P geometry (Angstrom): O, H1, H2
WATER_TEMPLATE_ANG = np.array([[0.0000, 0.0000, 0.0],
                               [0.9572, 0.0000, 0.0],
                               [-0.2400, 0.9266, 0.0]])


# Input
def load_inputs(psf_file, pdb_file, prm_files, default_box_nm=5.06):
    """Read PSF, PDB and CHARMM parameter files."""
    psf = app.CharmmPsfFile(psf_file)
    params = app.CharmmParameterSet(*prm_files)
    pdb = app.PDBFile(pdb_file)

    bv = pdb.topology.getPeriodicBoxVectors()
    if bv is not None:
        box_nm = np.array([bv[i][i].value_in_unit(unit.nanometer) for i in range(3)])
    else:
        box_nm = np.full(3, default_box_nm)
    psf.setBox(*(b * unit.nanometer for b in box_nm))
    return psf, params, pdb, box_nm


def classify_atoms(topology):
    """Peptide atoms, peptide heavy atoms and water molecules (O, H1, H2)."""
    pept_atoms = np.array([a.index for a in topology.atoms()
                           if a.residue.name.strip().upper() not in MM_RESIDUE_NAMES])
    pept_set = set(pept_atoms.tolist())
    pept_heavy = np.array([a.index for a in topology.atoms()
                           if a.index in pept_set and a.element.symbol != 'H'])

    water_mols = []
    n_water_res = 0
    for res in topology.residues():
        if res.name.strip().upper() in WATER_RES_NAMES:
            n_water_res += 1
            ats = list(res.atoms())
            O = [a.index for a in ats if a.element.symbol == 'O']
            H = [a.index for a in ats if a.element.symbol == 'H']
            if len(O) == 1 and len(H) == 2:
                water_mols.append((O[0], H[0], H[1]))
    if len(water_mols) != n_water_res:
        raise RuntimeError(f'{n_water_res - len(water_mols)} water residues do not '
                           'have the expected O/H/H composition')
    return pept_atoms, pept_heavy, water_mols


# MM system
def create_mm_system(psf, params, cutoff_nm=1.2, switch_nm=1.0, ewald_tol=1e-5):
    """Periodic CHARMM36m/TIP3P system (PME, X-H constraints, rigid water)."""
    system = psf.createSystem(params,
                              nonbondedMethod=app.PME,
                              nonbondedCutoff=cutoff_nm * unit.nanometer,
                              switchDistance=switch_nm * unit.nanometer,
                              constraints=app.HBonds,
                              rigidWater=True,
                              ewaldErrorTolerance=ewald_tol)
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            f.setUseSwitchingFunction(True)
            f.setSwitchingDistance(switch_nm * unit.nanometer)
            f.setUseDispersionCorrection(False)
    return system


def remove_peptide_internal_mm(system, pept_atoms, keep_bonded=False, verbose=True):
    """Remove MM interactions between peptide atoms (keep_bonded: nonbonded only)."""
    pept = set(int(i) for i in pept_atoms)
    removed = []

    def classify(idx):
        n_in = sum(i in pept for i in idx)
        if n_in == 0:
            return 'mm'
        if n_in == len(idx):
            return 'peptide'
        raise RuntimeError(f'unsupported bonded term: {idx}')

    bonded_types = (openmm.HarmonicBondForce, openmm.HarmonicAngleForce,
                    openmm.PeriodicTorsionForce, openmm.CustomTorsionForce,
                    openmm.CMAPTorsionForce)
    for f in system.getForces():
        name = type(f).__name__
        n = 0

        if keep_bonded and isinstance(f, bonded_types):
            continue

        if isinstance(f, openmm.HarmonicBondForce):
            for j in range(f.getNumBonds()):
                p1, p2, r0, k = f.getBondParameters(j)
                if classify((p1, p2)) == 'peptide':
                    f.setBondParameters(j, p1, p2, r0, 0.0)
                    n += 1

        elif isinstance(f, openmm.HarmonicAngleForce):
            for j in range(f.getNumAngles()):
                p1, p2, p3, a0, k = f.getAngleParameters(j)
                if classify((p1, p2, p3)) == 'peptide':
                    f.setAngleParameters(j, p1, p2, p3, a0, 0.0)
                    n += 1

        elif isinstance(f, openmm.PeriodicTorsionForce):
            for j in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = f.getTorsionParameters(j)
                if classify((p1, p2, p3, p4)) == 'peptide':
                    f.setTorsionParameters(j, p1, p2, p3, p4, per, phase, 0.0)
                    n += 1

        elif isinstance(f, openmm.CustomTorsionForce):
            names = [f.getPerTorsionParameterName(i)
                     for i in range(f.getNumPerTorsionParameters())]
            if 'k' not in names:
                raise RuntimeError('unsupported CustomTorsionForce')
            k_idx = names.index('k')
            for j in range(f.getNumTorsions()):
                p1, p2, p3, p4, pars = f.getTorsionParameters(j)
                if classify((p1, p2, p3, p4)) == 'peptide':
                    pars = list(pars)
                    pars[k_idx] = 0.0
                    f.setTorsionParameters(j, p1, p2, p3, p4, pars)
                    n += 1

        elif isinstance(f, openmm.CMAPTorsionForce):
            for j in range(f.getNumTorsions()):
                t = f.getTorsionParameters(j)
                if classify(t[1:]) != 'peptide':
                    raise RuntimeError('unsupported CMAP torsion')
            for j in range(f.getNumMaps()):
                size, vals = f.getMapParameters(j)
                f.setMapParameters(j, size, [0.0] * len(vals))
            n = f.getNumTorsions()

        elif isinstance(f, openmm.NonbondedForce):
            existing = {}
            for k in range(f.getNumExceptions()):
                p1, p2, qq, sig, eps = f.getExceptionParameters(k)
                if p1 in pept and p2 in pept:
                    existing[(min(p1, p2), max(p1, p2))] = k
            for (p1, p2), k in existing.items():
                _, _, qq, sig, eps = f.getExceptionParameters(k)
                f.setExceptionParameters(k, p1, p2, 0.0, sig, 0.0)
            plist = sorted(pept)
            for a in range(len(plist)):
                for b in range(a + 1, len(plist)):
                    if (plist[a], plist[b]) not in existing:
                        f.addException(plist[a], plist[b], 0.0, 1.0, 0.0)
            n = len(plist) * (len(plist) - 1) // 2

        elif isinstance(f, (openmm.CMMotionRemover, openmm.MonteCarloBarostat)):
            continue

        else:
            raise NotImplementedError(f'unsupported force: {name}')

        removed.append((name, n))

    return removed


def peptide_unwrap_tree(bonds, pept_atoms):
    """Bond spanning tree of the peptide (child, parent), used by make_whole."""
    pept = [int(a) for a in pept_atoms]
    pset = set(pept)
    nbr = {a: [] for a in pept}
    for i, j in bonds:
        i, j = int(i), int(j)
        if i in pset and j in pset:
            nbr[i].append(j)
            nbr[j].append(i)
    child, parent, seen, queue = [], [], {pept[0]}, [pept[0]]
    while queue:
        a = queue.pop(0)
        for b in nbr[a]:
            if b not in seen:
                seen.add(b); queue.append(b); child.append(b); parent.append(a)
    if len(seen) != len(pept):
        raise RuntimeError(f'peptide bond graph is not connected '
                           f'({len(seen)} of {len(pept)} atoms reached)')
    return np.array(child, dtype=int), np.array(parent, dtype=int)


def make_whole(pos, box, child, parent):
    """Make a molecule whole across periodic boundaries."""
    for c, p in zip(child, parent):
        pos[c] = pos[p] + pbc_diff(pos[c] - pos[p], box)
    return pos


# Geometry helpers
def pbc_diff(v, box):
    """Minimum-image displacement (orthorhombic box)."""
    return v - box * np.round(v / box)


def switching(r, r_on, r_off):
    """Switching weight: 1 for r <= r_on, 0 for r >= r_off."""
    r = np.asarray(r, dtype=np.float64)
    s = np.zeros_like(r)
    s[r <= r_on] = 1.0
    m = (r > r_on) & (r < r_off)
    s[m] = np.cos(0.5 * np.pi * (r[m] - r_on) / (r_off - r_on)) ** 2
    return s


def dummy_slot_offsets(n_slots, start_ang=200.0, spacing_ang=12.0):
    """Placeholder positions for unused water slots (Angstrom)."""
    off = np.zeros((n_slots, 3, 3))
    off[:, :, 0] = (start_ang + spacing_ang * np.arange(n_slots))[:, None]
    return off + WATER_TEMPLATE_ANG[None, :, :]
