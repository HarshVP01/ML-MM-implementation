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




# How to execute the program





# Reference
```
DOI for our preprint
```
