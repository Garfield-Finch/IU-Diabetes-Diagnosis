# Make System modifications to the Learning Health Network (LHN) VM

This document provides an overview of the setup process for the Learning Health Network (LHN) VM. The setup process is automated using the `setup_lhn.sh` script located in the `update_lhn` directory.

## Prerequisites

Ensure the following files are present in the `update_lhn` directory:
- `setup_lhn.sh`
- `install_miniconda.sh`
- `environment.yml`

## Running the Setup Script

To automate the setup process, run the `setup_lhn.sh` script:

```bash
sh update_lhn/setup_lhn.sh
```

After the script completes, you may need to source your `.bashrc` file to apply changes:

```bash
source ~/.bashrc
```

## Manual Steps

### Configure SSH Key for GitHub

The script will handle the SSH key generation and configuration. You will need to manually add the generated SSH key to your GitHub account under Settings > SSH and GPG keys > New SSH key.

### Clone the GitHub Repository

The script will clone the GitHub repository if it is not already present. Ensure you have SSH access configured correctly.

### Create the Conda Environment

The script will create the Conda environment using the `environment.yml` configuration file. If the environment already exists, it will skip this step.

### Running the Run Script

The script will make the `run.sh` script executable but will not run it automatically. You will need to run it manually after activating the Conda environment:

```bash
conda activate IUHealth
./run.sh
```

## Detailed Steps (For Reference)

The `setup_lhn.sh` script performs the following steps:

1. **Check and Create .ssh Directory**:
   - If the `.ssh` directory does not exist, it will be created.

2. **Generate SSH Key**:
   - If SSH keys do not exist, a new SSH key will be generated.
   - The SSH key will be added to the `ssh-agent`.
   - The public key will be saved to `./id_ed25519.pub.txt` for manual addition to GitHub.

3. **Verify SSH Connection to GitHub**:
   - The script will verify the SSH connection to GitHub.
   - You will be prompted to confirm that the SSH key has been added to GitHub.

4. **Install Miniconda**:
   - If Miniconda is not installed or the version is outdated, the script will install or update Miniconda using the `install_miniconda.sh` script.

5. **Clone the GitHub Repository**:
   - If the GitHub repository has not been cloned, the script will clone it.

6. **Create the Conda Environment**:
   - If the Conda environment does not exist, the script will create it using the `environment.yml` file.

7. **Make the Run Script Executable**:
   - The script will make the `run.sh` script executable but will not run it automatically.

## Additional Notes

- The LHN VM is created from a Docker script every day. Until the script is updated, we need to manually install the latest software.
- Ensure that `install_miniconda.sh`, `environment.yml`, and `setup_lhn.sh` are all in the `update_lhn` directory.
