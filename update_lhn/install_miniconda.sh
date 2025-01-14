#!/bin/sh

# Define file paths
LOCAL_FILE=~/work/Users/hnelson3/Miniconda3-latest-Linux-x86_64.sh
CHECKSUM_FILE=~/work/Users/hnelson3/Miniconda3-latest-Linux-x86_64.sh.sha256

# Function to download the checksum file
download_checksum() {
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh.sha256 -O "$CHECKSUM_FILE"
}

# Function to download the installer
download_installer() {
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$LOCAL_FILE"
}

# Function to calculate the checksum of the local file
calculate_local_checksum() {
  if [ -f "$LOCAL_FILE" ]; then
    sha256sum "$LOCAL_FILE" | awk '{ print $1 }'
  else
    echo ""
  fi
}

# Function to read the remote checksum
read_remote_checksum() {
  cat "$CHECKSUM_FILE" | awk '{ print $1 }'
}

# Download the checksum file
download_checksum

# Calculate the local checksum
LOCAL_CHECKSUM=$(calculate_local_checksum)

# Read the remote checksum
REMOTE_CHECKSUM=$(read_remote_checksum)

# Compare checksums and download the installer if they don't match
if [ "$LOCAL_CHECKSUM" != "$REMOTE_CHECKSUM" ]; then
  download_installer
fi

# Run the Miniconda installer
bash "$LOCAL_FILE"