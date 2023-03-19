# RL

Reinforecement Learning for program transformations.

### Graphs (and other large) files

```
https://cloud.pmf.uns.ac.rs/s/iF5n47F4fDMn6P9
```

### Requirements

1. Docker/Podman for Fermat image
2. Conda environment for Python dependencies (`conda-env.yml`, `conda env create -f conda-env.yml`)
3. Input wsl data (`hc-fit-mj-alpha-2020-06-final.tgz` -- see cloud folder above)
4. Output csv graph data is in `graphs.tar.xz`


### Verification of HCF (Hill Climb in FermaT) data

The Hillclimbing program (`hill_climbing.wsl`), and the translator from MicroJava compiled to WSL
(`mjc2wsl`) are available at

```
https://github.com/quinnuendo/mjc2wsl/
```

These can be used to verify the generated data given in this repository and
the above links.