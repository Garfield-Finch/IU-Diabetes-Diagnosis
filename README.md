# Make System modifications to the Learning Health Network (LHN) VM.

run update_lhn/setup_lhn.sh to do all of this in a shell script.
except for the last part of ./run.sh  Do that by hand after activating the environment

## Configure SSH key for GitHub

You will need to clone the repository from GitHub. You will need to configure SSH keys to do this.

### Check for .ssh directory

Create this directory only if you don't already have it.

```
ls -al ~/.ssh
```

```
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

## Check for SSH keys

Create keys only if needed.

```
ls -al ~/.ssh/id_*
```

## Generate a new SSH key

### First install `ssh-keygen` if needed.

```
sudo apt-get install openssh-client
```

- Note to replace `harlananelson@gmail.com` With your email

```
ssh-keygen -t ed25519 -C "harlananelson@gmail.com"
```

### Get ride of unneeded software.
```
sudo apt autoremove
```

### Add the SSH key to ssh-agent

```
eval "$(ssh-agent -s)" 
ssh-add ~/.ssh/id_ed25519
```

### You need to copy this key to github SSH key 

Alternate way to copy the key to the clipboard.

```
cat ~/.ssh/id_ed25519.pub > ~/work/Users/hnelson3/ssh_key.txt
echo "Your SSH key has been saved to ~/ssh_key.txt"
```
copy then delete that file

```
rm ~/work/Users/hnelson3/ssh_key.txt
```


# Put key in github SSH Keys

#### Remove old SSH keys if needed

rm ~/.ssh/id_ed25519
rm ~/.ssh/id_ed25519.pub
rm ~/ssh_key.txt



The LHN is created from a docker script every day.  Currently that script installs outdated software.  Until the script is updated, we need to manually install the latest software.

## Install  `microconda`.  

 
 Run the `install_miniconda.sh` script to install `miniconda`.

 ```
 sh update_lhn/install_miniconda.sh
 ```


### Step 2. Clone this GitHub repository

### Tell git who you are

```
git config --global user.email "harlananelson@gmail.com"
git config --global user.name "Harlan Nelson"
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

```bash
chmod +x run.sh
./run.sh
```

