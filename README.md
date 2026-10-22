# Acronym: [brainstorm]

# lib dependence
## lib dependence

Tested on Linux with NVIDIA V100 GPUs (NCI Gadi):

```
python    3.12.13
openmm    8.3.1      (CUDA platform)
numpy     2.4.6
jax       0.5.3      (GPU build)
jax-md    0.2.27
ase       3.29.0
so3lr     0.1.0
mdtraj    1.11.1     (optional: trajectory checks)
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
Please also cite the software this work builds on:

- **SO3LR**: A. Kabylda, J. T. Frank, S. Suárez-Dou, A. Khabibrakhmanov, L. Medrano Sandonas, O. T. Unke, S. Chmiela, K.-R. Müller, A. Tkatchenko, *J. Am. Chem. Soc.* **2025**, 147, 33723–33734. DOI: 10.1021/jacs.5c09558

Download PDF
- **OpenMM 8**: P. Eastman et al., *J. Phys. Chem. B* **2024**, 128, 109–116. DOI: 10.1021/acs.jpcb.3c06662

Download PDF
- **CHARMM36m**: J. Huang et al., *Nat. Methods* **2017**, 14, 71–73. DOI: 10.1038/nmeth.4067

Download PDF
- **ONIOM-XS** (adaptive buffer scheme): T. Kerdcharoen, K. Morokuma, *Chem. Phys. Lett.* **2002**, 355, 257–262. DOI: 10.1016/S0009-2614(02)00210-5
Download PDF
