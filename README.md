# Acronym: [brainstorm]

# lib dependence
```
python    3.12.13
openmm    8.3.1
numpy     2.4.6
jax       0.5.3
jax-md    0.2.27
ase       3.29.0
so3lr     0.1.0
```

Example installation:

```bash
conda create -n so3lrmm -c conda-forge python=3.12 openmm=8.3.1
conda activate so3lrmm
pip install numpy==2.4.6 ase==3.29.0 jax-md==0.2.27 mdtraj==1.11.1 "jax[cuda12]==0.5.3"
pip install git+https://github.com/general-molecular-simulations/so3lr.git
```

# How to prepare the input files

The program needs three kinds of input:

1. **CHARMM PSF** (`peptide.psf`) of the solvated system
2. **PDB** (`peptide.pdb`) with starting coordinates and a `CRYST1` record giving the box size
3. **CHARMM parameter files**: `par_all36m_prot.prm` and `par_water_ions_namd.prm` (CHARMM36m)

The PSF/PDB pair can be built with CHARMM-GUI (Solution Builder) or VMD psfgen.
Requirements:

- **One peptide** (a single bonded molecule). Every residue that is not water or an ion is treated as the peptide (ML region).
- **Water**: TIP3P, residue name `TIP3` (`HOH` and `WAT` are also recognised).
- **Ions** (e.g. `SOD`, `CLA`): always treated with MM. They never enter the ML region.
- **Box**: orthorhombic (cubic recommended), large enough that the shell (6 Å around the peptide) never meets its own periodic image.
- **Force field**: standard CHARMM36m terms only.

# How to execute the program
```bash
python adaptive_so3lr_mm.py --name peptide --replica 1 \
    --psf peptide.psf --pdb peptide.pdb \
    --prm par_all36m_prot.prm par_water_ions_namd.prm \
    --out peptide_rep1 --jax-cache ./jax_cache
```

The program runs the full protocol: SO3LR relaxation of the peptide, MM
minimisation and equilibration, NVT warm-up with a time-step ramp, NVT
equilibration (500 ps), NPT equilibration (500 ps) and NPT production
(15 ns, 1 fs, 300 K, 1 bar, Langevin friction 10 ps⁻¹, Monte Carlo barostat).

Main options (`python adaptive_so3lr_mm.py --help` lists all):

| Option | Default | |
|---|---|---|
| `--r-on`, `--r-off` | 3.5, 6.0 Å | inner and outer radius of the switching region |
| `--temperature` | 300 K | |
| `--pressure` | 1 bar | |
| `--friction` | 10 ps⁻¹ | Langevin friction |
| `--nvt-steps`, `--npt-steps` | 500 000, 500 000 | NVT and NPT equilibration steps (1 fs) |
| `--prod-steps` | 15 000 000 | production steps (1 fs) |
| `--save-every` | 1000 | trajectory frame interval (steps) |
| `--replica`, `--seed` | 1, 1000 × replica | independent replicas |

Outputs in `--out`:

| File | |
|---|---|
| `<name>_rep<r>_npt_equil.dcd` | NPT equilibration trajectory |
| `<name>_rep<r>_prod.dcd` | production trajectory |
| `<name>_rep<r>_prod.csv` | temperature, SO3LR energy, number of ML waters |
| `<name>_rep<r>_prod.chk` | OpenMM checkpoint |

# Reference
```
If you use this code, please cite:
DOI for our preprint
```
