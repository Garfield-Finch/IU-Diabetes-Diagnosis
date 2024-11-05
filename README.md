### Step 1. Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

### Step 2. Clone this GitHub repository

```bash
git clone https://github.com/Garfield-Finch/IU-Diabetes-Diagnosis.git
```

### Step 3. Create the conda environment using the configuration file in `config/environment.yml`  # TODO: to verify the file name

```bash
conda env create -f ./config/environment.yml
conda activate IUHealth
```
# TODO: To verify the conda env name

### Step 4. Run the code

```bash
python main.py -i ./dataset.csv
```
