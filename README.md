# Periodic SSH Chain – Master Thesis Valorization Codegit 

This repository contains numerical and analytical calculations 
for the periodic Su-Schrieffer-Heeger (SSH) model developed 
as part of my Master's thesis research at EPFL.

The project focuses on:

- Construction of the SSH Hamiltonian
- Implementation of periodic boundary conditions
- Eigenvalue spectrum computation
- Analytical comparison of numerical results
- Visualization of energy bands
- Concurrence Plots
- Entanglement Phase Transition

---

## Repository Structure

periodic_chain.ipynb   → Main notebook containing calculations and analysis  
environment.yml       → Conda environment configuration  
README.md             → Project description  

---

## Requirements

The project was developed using:

- Python 3.x
- NumPy
- SciPy
- Matplotlib

All dependencies are listed in `environment.yml`.

To recreate the environment locally: 

conda env create -f environment.yml
conda activate thesis_python


If you prefer using pip:

pip install -r requirements.txt


---

## Running on Google Colab

You can run this project directly on Google Colab.

Clone the repository:
!git clone https://github.com/
<<YOUR_GITHUB_USERNAME>>/<<YOUR_REPOSITORY_NAME>>.git
%cd <<YOUR_REPOSITORY_NAME>>



