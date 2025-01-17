#!/bin/bash

# Uninstall old version of R (if installed via Conda)
conda remove r-base r-essentials -y

# Update package lists and install necessary packages
sudo apt update -qq
sudo apt install --no-install-recommends software-properties-common dirmngr -y

# Add the CRAN repository key
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc

# Add the CRAN repository to the sources list
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/"

# Update package lists again
sudo apt update -qq

# Install R base and recommended packages
sudo apt install --no-install-recommends r-base r-base-dev -y

# Install IRkernel in R
R -e "install.packages('IRkernel', repos='https://cran.rstudio.com/')"

# Install IRkernel into Jupyter
R -e "IRkernel::installspec(user = FALSE)"

# Verify the installation
jupyter kernelspec list

echo "R kernel installation for Jupyter is complete."