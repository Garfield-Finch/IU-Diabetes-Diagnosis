### Step 1. Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

#### or use Miniconda

```
cd 
```
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/work/Users/hnelson3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/hnelson3/miniconda3/miniconda.sh
```

### Step 2. Clone this GitHub repository

#### With https

```bash
git clone https://github.com/Garfield-Finch/IU-Diabetes-Diagnosis.git
```
#### With ssh

```bash
git clone git@github.com:Garfield-Finch/IU-Diabetes-Diagnosis.git
```

### Step 3. Create the conda environment using the configuration file in `environment.yml`

```bash
conda env create -f environment.yml
conda activate IUHealth
```

### Step 4. Run the code

```bash
chmod +x run.sh
./run.sh
```
### Reverence

#### Configure SSH key for GitHub

##### If you don't have the .ssh directory.

mkdir -p ~/.ssh
chmod 700 ~/.ssh

#### Generate a new SSH key

- Note to replace with your email
ssh-keygen -t ed25519 -C "harlananelson@gmail.com"

#### Add the SSH key to ssh-agent
eval "$(ssh-agent -s)" 
ssh-add ~/.ssh/id_ed25519

#### You need to copy this key to github SSH key 

cat ~/.ssh/id_ed25519.pub > ~/ssh_key.txt
echo "Your SSH key has been saved to ~/ssh_key.txt"

# Put key in github SSH Keys

#### Remove old SSH keys if needed

rm ~/.ssh/id_ed25519
rm ~/.ssh/id_ed25519.pub
rm ~/ssh_key.txt
