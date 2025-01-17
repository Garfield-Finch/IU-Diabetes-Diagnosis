#!/bin/bash

# Update package lists and install necessary packages
sudo apt update -qq
sudo apt install --no-install-recommends software-properties-common dirmngr -y

# Add the CRAN repository key
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc

# Add the CRAN repository to the sources list
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/"

# Add the universe repository
sudo add-apt-repository universe

# Update package lists again
sudo apt update -qq

# Install missing dependencies
sudo apt install -y pkgconf

# Manually download and install libdeflate-dev if not available
if ! sudo apt install -y libdeflate-dev; then
    wget http://archive.ubuntu.com/ubuntu/pool/main/libd/libdeflate/libdeflate-dev_1.0-2_amd64.deb
    sudo dpkg -i libdeflate-dev_1.0-2_amd64.deb
    sudo apt-get install -f  # Fix any broken dependencies
fi

# Install R base and recommended packages
sudo apt install --no-install-recommends r-base r-base-dev -y

# Clean up
sudo apt-get autoremove -y
sudo apt-get clean

echo "R installation/update is complete."