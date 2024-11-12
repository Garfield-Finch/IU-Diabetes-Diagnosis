### Step 1. Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

### Step 2. Clone this GitHub repository

```bash
git clone https://github.com/Garfield-Finch/IU-Diabetes-Diagnosis.git
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
