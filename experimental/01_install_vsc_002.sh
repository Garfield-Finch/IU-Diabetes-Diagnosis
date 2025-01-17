#!/bin/bash

# Define file paths and directories
WORK_DIR=~/work/Users/hnelson3
PROJECT_DIR=$(pwd)

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python 3.11
install_python311() {
    echo "Installing Python 3.11..."

    # Create a temporary directory for Python installation
    mkdir -p ~/python_install
    cd ~/python_install

    # Download and install Python 3.11
    wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz
    tar xzf Python-3.11.8.tgz
    cd Python-3.11.8

    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev

    # Configure and install Python
    ./configure --enable-optimizations --prefix=$HOME/.local
    make -j $(nproc)
    make altinstall

    # Create symlinks
    ln -sf $HOME/.local/bin/python3.11 $HOME/.local/bin/python3
    ln -sf $HOME/.local/bin/python3.11 $HOME/.local/bin/python

    # Clean up
    cd ~/
    rm -rf ~/python_install

    # Verify Python installation
    $HOME/.local/bin/python3.11 --version
}

# Function to install uv
install_uv() {
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH immediately
    export PATH="$HOME/.local/bin:$PATH"

    # Source the environment file
    if [ -f "$HOME/.local/bin/env" ]; then
        . "$HOME/.local/bin/env"
    fi

    # Verify uv installation
    if command_exists uv; then
        echo "uv installed successfully!"
        uv --version
    else
        echo "Error: uv not found in PATH after installation"
        echo "Current PATH: $PATH"
        exit 1
    fi
}

# Function to setup Python environment using uv
setup_with_uv() {
    echo "Setting up Python environment with uv..."

    # Ensure we're in the project directory
    cd "$PROJECT_DIR"

    # Remove existing venv if it exists
    rm -rf .venv

    echo "Creating virtual environment with Python 3.11..."
    $HOME/.local/bin/python3.11 -m venv .venv

    # Verify venv creation
    if [ ! -d ".venv" ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi

    echo "Activating virtual environment..."
    . .venv/bin/activate

    echo "Installing base packages with uv..."
    $HOME/.local/bin/uv pip install \
        jupyter \
        notebook \
        ipykernel \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        torch \
        torchvision \
        torchaudio \
        tensorboard \
        wandb \
        pyyaml \
        tqdm \
        requests

    # Conditionally install pyspark
    if [ "$INCLUDE_PYSPARK" == "y" ]; then
        echo "Installing pyspark..."
        $HOME/.local/bin/uv pip install \
            pyspark \
            py4j
    fi

    # Conditionally install R
    if [ "$INCLUDE_R" == "y" ]; then
        echo "Installing R and IRkernel..."
        # Install R
        sudo apt-get install -y r-base r-base-dev
        # Install IRkernel in R
        R -e "install.packages('IRkernel', repos='https://cran.rstudio.com/')"
        # Install the R kernel into Jupyter
        R -e "IRkernel::installspec(user = TRUE)"
    fi

    echo "Package installation completed"
}

# Main script
echo "Choose your installation method:"
echo "1) Use uv (modern, faster package manager)"
echo "2) Use conda (traditional approach)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        install_python311
        install_uv

        # Ask user if they want to include pyspark
        read -p "Do you want to include pyspark? (y/n): " INCLUDE_PYSPARK
        # Ask user if they want to include R
        read -p "Do you want to include R? (y/n): " INCLUDE_R

        setup_with_uv

        # Update bashrc
        cat >> ~/.bashrc << EOL

# Python and UV setup
export PATH="\$HOME/.local/bin:\$PATH"
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    . "$PROJECT_DIR/.venv/bin/activate"
fi
EOL

        # Setup Jupyter kernel
        echo "Setting up Jupyter kernel..."
        if [ -f ".venv/bin/python" ]; then
            .venv/bin/python -m ipykernel install --user --name python311-venv --display-name "Python 3.11 (venv)"
        else
            echo "Error: Virtual environment Python not found"
        fi
        ;;
    2)
        echo "Conda installation option not implemented yet"
        exit 1
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "\nInstallation complete! Please run:"
echo "source ~/.bashrc"
echo "cd $PROJECT_DIR && source .venv/bin/activate"

# Print verification steps
echo "\nTo verify installation:"
echo "1. which python"
echo "2. python --version"
echo "3. which uv"
echo "4. jupyter kernelspec list"