#!/bin/bash

# Create a new Conda environment
conda create -n r_env python=3.9 r-base r-essentials jupyter -y

# Activate the environment
source activate r_env

# Install IRkernel in R
R -e "install.packages('IRkernel', repos='https://cran.rstudio.com/')"

# Install the R kernel into Jupyter
R -e "IRkernel::installspec(user = TRUE)"

# Install ipykernel package
conda install ipykernel -y

# Create a new Python kernel
python -m ipykernel install --user --name r_env --display-name "Python (r_env)"

# Verify the installation
jupyter kernelspec list

echo "R and Python kernels installation for Jupyter is complete."