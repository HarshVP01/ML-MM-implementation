"""MM forces within the ML region (peptide + ML waters)."""

import numpy as np
import openmm
import openmm.unit as unit

KJ_MOL_NM = unit.kilojoules_per_mole / unit.nanometer

BONDED_TYPES = (openmm.HarmonicBondForce, openmm.HarmonicAngleForce,
                openmm.PeriodicTorsionForce, openmm.CustomTorsionForce,
                openmm.CMAPTorsionForce)


def copy_peptide_bonded(main_system, pept_atoms, target):
    """Copy the peptide bonded terms of main_system into target."""
    loc = {int(a): k for k, a in enumerate(pept_atoms)}

    def inside(*idx):
        return all(int(i) in loc for i in idx)

    out = []
    for f in main_system.getForces():
        if not isinstance(f, BONDED_TYPES):
            continue
        name = type(f).__name__
        n = 0
        if isinstance(f, openmm.HarmonicBondForce):
            nf = openmm.HarmonicBondForce()
            for j in range(f.getNumBonds()):
                a, b, r0, k = f.getBondParameters(j)
                if inside(a, b):
                    nf.addBond(loc[a], loc[b], r0, k); n += 1
        elif isinstance(f, openmm.HarmonicAngleForce):
            nf = openmm.HarmonicAngleForce()
            for j in range(f.getNumAngles()):
                a, b, c, t0, k = f.getAngleParameters(j)
                if inside(a, b, c):
                    nf.addAngle(loc[a], loc[b], loc[c], t0, k); n += 1
        elif isinstance(f, openmm.PeriodicTorsionForce):
            nf = openmm.PeriodicTorsionForce()
            for j in range(f.getNumTorsions()):
                a, b, c, d, per, ph, k = f.getTorsionParameters(j)
                if inside(a, b, c, d):
                    nf.addTorsion(loc[a], loc[b], loc[c], loc[d], per, ph, k); n += 1
        elif isinstance(f, openmm.CustomTorsionForce):
            nf = openmm.CustomTorsionForce(f.getEnergyFunction())
            for j in range(f.getNumGlobalParameters()):
                nf.addGlobalParameter(f.getGlobalParameterName(j),
                                      f.getGlobalParameterDefaultValue(j))
            for j in range(f.getNumPerTorsionParameters()):
                nf.addPerTorsionParameter(f.getPerTorsionParameterName(j))
            for j in range(f.getNumTorsions()):
                a, b, c, d, pars = f.getTorsionParameters(j)
                if inside(a, b, c, d):
                    nf.addTorsion(loc[a], loc[b], loc[c], loc[d], pars); n += 1
        else:                                            # CMAPTorsionForce
            nf = openmm.CMAPTorsionForce()
            for j in range(f.getNumMaps()):
                nf.addMap(*f.getMapParameters(j))
            for j in range(f.getNumTorsions()):
                m, *at = f.getTorsionParameters(j)
                if inside(*at):
                    nf.addTorsion(m, *[loc[a] for a in at]); n += 1
                elif any(int(a) in loc for a in at):
                    raise RuntimeError('unsupported CMAP torsion')
        nf.setForceGroup(f.getForceGroup())
        target.addForce(nf)
        out.append((name, n))
    return out


class InnerMMContext:
    """OpenMM context for the MM forces inside the ML region."""

    def __init__(self, main_system, pept_atoms, water_mols, n_slots,
                 platform, platform_props=None):
        forces = main_system.getForces()
        if any(isinstance(f, openmm.CustomNonbondedForce) for f in forces):
            raise NotImplementedError('unsupported force: CustomNonbondedForce')
        nbs = [f for f in forces if isinstance(f, openmm.NonbondedForce)]
        if len(nbs) != 1:
            raise RuntimeError('expected exactly one NonbondedForce in the main system')
        nb_main = nbs[0]

        self.pept_atoms = np.asarray(pept_atoms, dtype=int)
        self.n_pept = len(self.pept_atoms)
        self.water_mols = water_mols
        self.n_slots = n_slots
        self.n_inner = self.n_pept + 3 * n_slots

        n_main = nb_main.getNumParticles()
        self._q = np.zeros(n_main)
        self._sig = np.zeros(n_main)
        self._eps = np.zeros(n_main)
        for i in range(n_main):
            q, sig, eps = nb_main.getParticleParameters(i)
            self._q[i] = q.value_in_unit(unit.elementary_charge)
            self._sig[i] = sig.value_in_unit(unit.nanometer)
            self._eps[i] = eps.value_in_unit(unit.kilojoules_per_mole)

        system = openmm.System()
        for _ in range(self.n_inner):
            system.addParticle(1.0)

        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        nb.setUseSwitchingFunction(False)
        nb.setUseDispersionCorrection(False)

        for idx in self.pept_atoms:
            nb.addParticle(self._q[idx], self._sig[idx], self._eps[idx])
        for _ in range(3 * n_slots):
            nb.addParticle(0.0, 0.1, 0.0)

        for a in range(self.n_pept):
            for b in range(a + 1, self.n_pept):
                nb.addException(a, b, 0.0, 1.0, 0.0)
        for s in range(n_slots):
            o = self.n_pept + 3 * s
            for p1, p2 in ((o, o + 1), (o, o + 2), (o + 1, o + 2)):
                nb.addException(p1, p2, 0.0, 1.0, 0.0)

        nb.setForceGroup(nb_main.getForceGroup())
        system.addForce(nb)
        self._nb = nb
        self._system = system

        self.bonded_forces = copy_peptide_bonded(main_system, self.pept_atoms, system)

        props = dict(platform_props or {})
        if platform.getName() == 'CUDA':
            props['DeterministicForces'] = 'true'
        self._integrator = openmm.VerletIntegrator(0.001)
        self._context = openmm.Context(system, self._integrator, platform, props)

        self._slot_mol = np.full(n_slots, -1)

    def set_slots(self, slot_mol):
        """Assign water molecules (index or -1) to slots."""
        slot_mol = np.asarray(slot_mol)
        changed = np.nonzero(slot_mol != self._slot_mol)[0]
        if changed.size == 0:
            return
        for s in changed:
            base = self.n_pept + 3 * s
            m = slot_mol[s]
            if m >= 0:
                for k, atom in enumerate(self.water_mols[m]):
                    self._nb.setParticleParameters(base + k, self._q[atom],
                                                   self._sig[atom], self._eps[atom])
            else:
                for k in range(3):
                    self._nb.setParticleParameters(base + k, 0.0, 0.1, 0.0)
        self._nb.updateParametersInContext(self._context)
        self._slot_mol = slot_mol.copy()

    def evaluate(self, pept_pos_ang, slot_pos_ang):
        """Positions (Angstrom) -> forces on peptide and slots (kJ/mol/nm)."""
        pos_nm = np.concatenate([np.asarray(pept_pos_ang, dtype=np.float64),
                                 np.asarray(slot_pos_ang, dtype=np.float64).reshape(-1, 3)]) / 10.0
        self._context.setPositions(pos_nm)
        F = self._context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(KJ_MOL_NM)
        F = np.asarray(F)
        return F[:self.n_pept], F[self.n_pept:].reshape(self.n_slots, 3, 3)
