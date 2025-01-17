#!/bin/sh

# Define file paths
LOCAL_FILE=~/work/Users/hnelson3/Miniconda3-latest-Linux-x86_64.sh
CHECKSUM_FILE=~/work/Users/hnelson3/Miniconda3-latest-Linux-x86_64.sh.sha256

# Official SHA-256 hash value for the latest Miniconda installer
# You need to get this from the Anaconda website or the official documentation
OFFICIAL_SHA256="<OFFICIAL_SHA256_HASH>"

# Function to download the installer
download_installer() {
  curl -L -o "$LOCAL_FILE" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
}

# Function to calculate the checksum of the local file
calculate_local_checksum() {
  if [ -f "$LOCAL_FILE" ]; then
    sha256sum "$LOCAL_FILE" | awk '{ print $1 }'
  else
    echo ""
  fi
}

# Download the installer
download_installer

# Calculate the local checksum
LOCAL_CHECKSUM=$(calculate_local_checksum)

# Compare checksums
if [ "$LOCAL_CHECKSUM" != "$OFFICIAL_SHA256" ]; then
  echo "Checksum verification failed! The downloaded file may be corrupted or tampered with."
  echo "Expected: $OFFICIAL_SHA256"
  echo "Got: $LOCAL_CHECKSUM"
  exit 1
fi

# Run the Miniconda installer
bash "$LOCAL_FILE"