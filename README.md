## Modelling a gravity-mode pulsator

The neural network for Computing Pulsation Periods and Photospheric Observables (`C-3PO`) is trained on a grid of MESA stellar evolution models and GYRE pulsation models, covering the entire γ Doradus instability strip. `C-3PO` is integrated into a forward modelling scheme designed by [Mombarg et al. (2021)](https://www.aanda.org/articles/aa/pdf/2021/06/aa39543-20.pdf) to estimate the stellar mass and age from the observed prograde dipole gravity-mode pulsations, and optionally also the luminosity, effective temperature, and surface gravity. The `C-3PO` network can be downloaded [here](https://doi.org/10.5281/zenodo.6925830).


### Prerequisites
To run `C-3PO`, you need the `keras-enviroment.yml` and `requirements.txt` file. Then, run the following commands (make sure you have Conda installed first):

```markdown
conda env create -f keras-environment.yml
conda activate keras-env
pip install -r ./requirements.txt
```

You can then simply run `Model_star.py` once you have set all the correct paths in this file. The neural network modules are relatively large. If your system cannot handle this, you can try to load only a few of the 
`C-3PO_Zext_npg15-91_P_*.h5` modules.

An online tutorial on asteroseismic modelling with `C-3PO` that I gave at the TASC6/KASC13 conference can be found [here](https://www.youtube.com/watch?v=0WBb-qwNsWs).
