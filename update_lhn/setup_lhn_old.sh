#!/bin/bash

# README

# This will set up ssh
# Generate and SSH key
# install miniconda
# clone the repo
# use conda to set up the conda environment

# Make sure setup_lhn.sh  (this)
#           install_miniconda.sh
#           environment.yml
# Are all in the directory you use to
# sh setup_lhn.sh
# when finished run
# source ~/.bashrc
#
# Set up variables
EMAIL="harlananelson@example.com"
USER_NAME="Harlan A Nelson"
GITHUB_REPO="git@github.com:Garfield-Finch/IU-Diabetes-Diagnosis.git"
MINICONDA_SCRIPT="./install_miniconda.sh"
ENV_YAML="./environment.yml"  # Assuming environment.yml is one directory up
MINICONDA_VERSION='4.12.0'
RUN_SCRIPT=false
INSTALL_MINICONDA=false

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Ensure the PATH includes necessary directories
export PATH="/usr/bin:/usr/local/bin:$PATH"

# Step 1: Install necessary SSH packages
if ! command_exists ssh-keygen; then
    echo "ssh-keygen is not installed: installing ssh-keygen"
    sudo apt-get update -qq
    sudo apt-get install -qq -y openssh-client
else
    echo "ssh-keygen is already installed: $(command -v ssh-keygen)"
fi

if ! command_exists ssh-add; then
    echo "ssh-add is not installed: installing ssh-add"
    sudo apt-get update -qq
    sudo apt-get install -qq -y openssh-client
else
    echo "ssh-add is already installed: $(command -v ssh-add)"
fi

if ! command_exists ssh-agent; then
    echo "ssh-agent is not installed: installing ssh-agent"
    sudo apt-get update -qq
    sudo apt-get install -qq -y openssh-client
else
    echo "ssh-agent is already installed: $(command -v ssh-agent)"
fi

# Step 2: Check for .ssh directory
if [ ! -d ~/.ssh ]; then
    echo "A .ssh directory does not exist: creating .ssh directory"
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
else
    echo ".ssh directory already exists"
fi

# Step 3: Check for existing SSH keys
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "SSH keys do not exist: Generating new SSH key"
    ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -N ""

    # Add the SSH key to ssh-agent
    echo "Adding SSH key to ssh-agent"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519

    # Copy the SSH key to a file for manual addition to GitHub
    cat ~/.ssh/id_ed25519.pub > ./id_ed25519.pub.txt
    echo "Your SSH key has been saved to ./id_ed25519.pub.txt"
    echo "Please add this key to your GitHub account under Settings > SSH and GPG keys > New SSH key."
else
    echo "SSH key already exists"
fi

echo "Press Enter after adding the SSH key to GitHub..."
read -p "Have you added the SSH key to GitHub? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Please add the SSH key to GitHub"
    CONNECT_GITHUB=false
else
    CONNECT_GITHUB=true
fi

# Step 4: Attempt to connect to GitHub and capture the output
if [ "$CONNECT_GITHUB" = true ]; then
    OUTPUT=$(ssh -T git@github.com 2>&1)

    # Capture the exit status
    EXIT_STATUS=$?

    # Print the output and exit status for debugging
    echo "SSH command output: $OUTPUT"
    echo "SSH command exit status: $EXIT_STATUS"

    # Check the output message for successful authentication
    if echo "$OUTPUT" | grep -q "You've successfully authenticated"; then
        echo "SSH connection successful with message: $OUTPUT"
    else
        echo "SSH connection failed with exit status [$EXIT_STATUS] and message: $OUTPUT"
    fi
fi

# Step 5: Install miniconda if not already installed or version is outdated
if command_exists conda; then
    echo "Miniconda is already installed"
    # Check the version of Miniconda
    CURRENT_VERSION=$(conda --version | awk '{print $2}')
    echo "Current Miniconda version: $CURRENT_VERSION"

    # Compare the current version with the desired version
    if [ "$(printf '%s\n' "$MINICONDA_VERSION" "$CURRENT_VERSION" | sort -V | head -n1)" = "$MINICONDA_VERSION" ]; then
        echo "Miniconda version $CURRENT_VERSION is up to date"
    else
        echo "Miniconda version $CURRENT_VERSION is older than the desired version $MINICONDA_VERSION"
        INSTALL_MINICONDA=true
    fi
else
    INSTALL_MINICONDA=true
fi

if [ "$INSTALL_MINICONDA" = true ]; then
    if [ -f "$MINICONDA_SCRIPT" ]; then
        echo "Installing Miniconda using $MINICONDA_SCRIPT"
        yes | sh "$MINICONDA_SCRIPT" -u
    else
        echo "Miniconda installation script not found at $MINICONDA_SCRIPT"
    fi
fi

# Check if the current directory is a Git repository
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    CLONE=false
    echo "This is a Git repository."
else
    CLONE=true
    echo "This is not a Git repository."
fi

if [ "$CLONE" = true ]; then

   # Step 6: Clone the GitHub repository
   echo "Cloning the GitHub repository"
   git config --global user.email "$EMAIL"
   git config --global user.name "$USER_NAME"
   git clone "$GITHUB_REPO"
   cd IU-Diabetes-Diagnosis
fi

# Step 7: Update Python in the base environment
if [ "$INSTALL_MINICONDA" = true ]; then
    echo "Updating Python to the latest version"
    conda install python=3.10 -y
fi

ENV_NAME=$(grep 'name:' "$ENV_YAML" | cut -d ' ' -f 2)
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' exists."
    ENV_NAME_EXISTS=true
else
    echo "Conda environment '$ENV_NAME' does not exist."
    ENV_NAME_EXISTS=false
fi

# Step 8: Create the Conda environment
if [ "$ENV_NAME_EXISTS" = false ]; then

   echo "Creating the Conda environment"

   conda env create -f "$ENV_YAML"
   source activate IUHealth
fi

if [ "$RUN_SCRIPT" = true ]; then
   # Step 9: Make the run script executable and run it
   echo "Running the run script"
   chmod +x run.sh
   ./run.sh
fi

# Source the .bashrc file to apply changes
# source ~/.bashrc

echo "Setup complete!"