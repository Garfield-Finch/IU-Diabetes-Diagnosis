#!/bin/bash

# Start the shell script with the shebang line, #!/bin/bash, to tell the system that the script should be executed with the Bash shell.

cd ~/work/Users/hnelson3

# Export the path to the Python package
export PYTHONPATH=$PYTHONPATH:~/work/Users/hnelson3

# Install Python packages
python -m pip install patsy numpy plotnine py4j
python -m pip install --upgrade scipy
python -m pip install lifelines geopandas

# Fetch the URL for the latest release
RELEASE_URL=$(curl -s "https://api.github.com/repos/quarto-dev/quarto-cli/releases" | grep "browser_download_url.*quarto.*linux-amd64.tar.gz" | head -n 1 | cut -d '"' -f 4)

# Extract the version number from the URL
QUARTO_VERSION=$(basename "${RELEASE_URL}" | sed -r 's/quarto-(.*)-linux-amd64.tar.gz/\1/')

# Create the directory for the specific version
sudo mkdir -p "/opt/quarto/${QUARTO_VERSION}"

# Download the latest release
sudo curl -o quarto.tar.gz -L "${RELEASE_URL}"
sudo tar -zxvf quarto.tar.gz \
    -C "/opt/quarto/${QUARTO_VERSION}" \
    --strip-components=1
sudo rm quarto.tar.gz

# Create a symbolic link for quarto
sudo ln -s /opt/quarto/${QUARTO_VERSION}/bin/quarto /usr/local/bin/quarto

# Install additional Python packages
python -m pip install jupyterlab-quarto duckdb pandas polars

# Add some system configuration
echo 'export SICKLE_CELL_DATA_PATH="/home/hnelson3/work/Users/hnelson3"' >> ~/.bashrc
source ~/.bashrc