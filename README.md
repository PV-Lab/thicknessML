# _thicknessML_
This sofware package implements _thicknessML_ framework that rapidly extracts/predicts semiconductor thin film thickness _d_ from optical spectroscopic reflection _R_ and transmission _T_. This repository is organized according to the two-stage transfer learning workflow in the _thicknessML_ framework.
1. **pre-training**: pre-training models using the generic simulated single Tauc-Lorentz (TL) dataset (in `pre-training.py`)
2. **transfer learning** or retraining: retrain the pre-trained models using the simulated literature perovskite dataset, and apply the retrained models on experimentally measured _R_ and _T_ spectra of six synthesized methylammonium lead iodide (MAPbI<sub>3</sub>) perovskite film (in `transfer-learning.py`)

The following paper describes the details of the _thickness_ framework: **_Transfer Learning for Rapid Extraction of Thickness from Optical Spectra of Semiconductor Thin Films_** <sub>(link to be added)</sub>
# Table of Contents
- [_thicknessML_](#thicknessML)
- [How to Cite](#how-to-cite)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

# How to Cite
Please cite the following work if you want to use _thicknessML_
```
To be added
```
# Installation
To install, clone the repository, navigate to the folder, and use `pip install -r requirements.txt` in a python 3.6 environment.

## A possible way to create a python 3.6 enviroment with conda[^1]
[Conda](https://conda.io/en/latest/) is the package manager that the [Anaconda](https://docs.continuum.io/anaconda/) distribution is built upon. It is a package manager that is both cross-platform and language agnostic (it can play a similar role to a pip and virtualenv combination).

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) allows you to create a minimal self contained Python installation, and then use the [Conda](https://conda.io/en/latest/) command to install additional packages.

First you will need [Conda](https://conda.io/en/latest/) to be installed and downloading and running the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) will do this for you. The installer [can be found here](https://docs.conda.io/en/latest/miniconda.html)

The next step is to create a new conda environment. A conda environment is like a virtualenv that allows you to specify a specific version of Python and set of libraries. Run the following commands from a terminal window:
```
conda create -n thicknessML python=3.6
```
This will create a minimal environment with only Python installed in it. To put your self inside this environment run:
```
source activate thicknessML
```
or

```
conda activate thicknessML
```

Now you're ready to run `pip install -r requirements.txt` once navigated into the cloned/downloaded thicknessML folder.

After the once-for-all installation, every time before running the code as described in [Usage](#usage),
 simply activate the installed environment by `source/conda activate thicknessML`.


# Usage

## Stage 1 `pre-training.py`
default: training **M**ulti**T**ask **L**earning models
```
python pre-training.py
```
if training **S**ingle-**T**ask **L**earning models
```
python pre-training.py --STL
```
## Stage 2 `transfer-learning.py`
default: loading MTL pre-trained models with partial-weight retraining
```
python transfer-learning.py
```
if doing full-weight retraining
```
python transfer-learning.py --full-weight
``` 
_loading STL pre-trained models is toggled by adding `--STL` as in Stage 1_

## Another way to obtain identical datasets
1. Download compressed data file `data.tar.gz` from [https://doi.org/10.6084/m9.figshare.23501715.v1](https://doi.org/10.6084/m9.figshare.23501715.v1)
2. Move `data.tar.gz` to inside the data directory.
3. Run `tar -xvf data.tar.gz` after navigating into the data directory.

Datasets will automatically appear in folder [data](./data/) after uncompressing using `tar`.

**Note that `.h5` files are quite reliant on specific `h5py` version. Please make sure to have `h5py` of `2.10.0` for smooth opening of the data files.**

## Included scripts and folders:

| Scripts | Description |
| ------------- | ------------------------------ |
| `pre-training.py`      | Stage 1: pre-train and save models on the TL dataset|
| `transfer-learning.py`      | Stage 2: retrain (transfer) pre-trained models on the literature perovskite dataset, and predict experimental perovskite film thicknesses from measured _RT_|
| `utils.py` | Auxiliary functions|

| Folders | Description |
| ------------- | ------------------------------ |
| [data](./data)  | hosts saved datasets; the [utils](./data/utils/) folder within also contains scripts for Tauc-Lorentz oscillator `TaucLorentz.py` and transfer-matrix method `ScatteringMatrix.py`. |
| [pre-trained models](./pre-trained%20models/) | hosts pre-trained models; pre-trained models outputted by a running of `pre-training.py` will replace the current saved pre-trained models.

# Authors
The code was primarily written by Siyu Isaac Parker Tian and Zhe Liu, under the supervision of Zhe Liu, Tonio Buonassisi and Qianxiao Li.

[^1]: explanations of this section borrow from https://pandas.pydata.org/docs/getting_started/install.html
