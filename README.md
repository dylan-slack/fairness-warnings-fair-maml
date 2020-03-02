# Fairness Warnings & Fair-MAML: Learning Fairly from Minimal Data

This is the code for our paper, "Fairness Warnings & Fair-MAML: Learning Fairly from Minimal Data."

Checkout the full [paper](https://arxiv.org/abs/1908.09092).

## Getting started

Setup virtual environment and install requirements:

```
conda create -n fairwarnmaml python=3.7
source activate fairwarnmaml
pip install -r requirements.txt
```

To run Fairness Warnings, you'll need a full version of CPLEX.  For students and faculty members, install a free full version using the instructions from [here](https://github.com/ustunb/slim-python).

Run `compas_warning_example.py` to generate a fairness warning for the COMPAS data set.  Detailed read outs of the SLIM results will be given in a folder called `./SLIMLOGS` that will be created on run.

Run `fair_maml_cc_example.py` to sweep over a range of gammas for Fair-MAML on the communities and crime task. 

## References

Please consider citing our paper if you found this work useful!

```
@article{Slack2019FairWarningsFairMAML},
	title={Fairness Warnings and Fair-MAML: Learning Fairly with Minimal Data},
	author={Dylan Slack and Sorelle Friedler and Emile Givental.},
	journal={Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)},
	year={2020},
}
```

## Citations

The fair-MAML code in `fair_maml_cc_example.py` is modified from a MAML implementation from github user [Jakie Loong](https://github.com/dragen1860) found in [this repository](https://github.com/dragen1860/MAML-Pytorch). Please check out the original implementation as its quite elegant!