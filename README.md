# Acronym: [brainstorm]

# lib dependence
## lib dependence

Tested on Linux with NVIDIA V100 GPUs (NCI Gadi):

```
python    3.12.13
openmm    8.3.1
numpy     2.4.6
jax       0.5.3
jax-md    0.2.27
ase       3.29.0
so3lr     0.1.0
mdtraj    1.11.1
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

1. **CHARMM PSF** (`GTG.psf`) of the solvated system
2. **PDB** (`GTG.pdb`) with starting coordinates and a `CRYST1` record giving the box size
3. **CHARMM parameter files**: `par_all36m_prot.prm` and `par_water_ions_namd.prm` (CHARMM36m)

The PSF/PDB pair can be built with CHARMM-GUI (Solution Builder) or VMD psfgen.
Requirements:

- **One peptide** (a single bonded molecule). Every residue that is not water or an ion is treated as the peptide (ML region).
- **Water**: TIP3P, residue name `TIP3` (`HOH` and `WAT` are also recognised).
- **Ions** (e.g. `SOD`, `CLA`): always treated with MM. They never enter the ML region.
- **Box**: orthorhombic (cubic recommended), large enough that the shell (6 Å around the peptide) never meets its own periodic image.
- **Force field**: standard CHARMM36m terms only.

# How to execute the program





# Reference
```
If you use this code, please cite:
DOI for our preprint
```
