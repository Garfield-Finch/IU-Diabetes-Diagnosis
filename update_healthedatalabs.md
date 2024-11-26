# Updating Python and Spark in a Docker Shell

This guide provides step-by-step instructions to update Python using Miniconda, install and update Spark, and configure a Jupyter kernel to use the updated environment within a Docker shell.

## Step 1: Install Miniconda and Update Python

1. **Download and Install Miniconda**:
   ```sh
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p ~/miniconda3
   ```

2. **Initialize Conda**:
   ```sh
   ~/miniconda3/bin/conda init
   source ~/.bashrc
   ```

3. **Create a New Conda Environment with Updated Python**:
   ```sh
   conda create --name myenv python=3.9 -y
   conda activate myenv
   ```

## Step 2: Install and Update Spark

1. **Install Spark**:
   ```sh
   conda install pyspark -y
   ```

2. **Update Spark** (if needed):
   ```sh
   conda install pyspark=3.2.0 -y
   ```

## Step 3: Configure Jupyter Kernel

1. **Install Jupyter**:
   ```sh
   conda install jupyter -y
   ```

2. **Create a Jupyter Kernel**:
   ```sh
   python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
   ```

## Step 4: Configure PySpark in Jupyter

1. **Set Up PySpark**:
   ```sh
   export SPARK_HOME=~/miniconda3/envs/myenv/lib/python3.9/site-packages/pyspark
   export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
   ```

2. **Verify PySpark**:
   - Start Jupyter and verify that PySpark is working correctly.
   ```sh
   jupyter notebook
   ```

   In a new Jupyter notebook, run the following code to verify PySpark:
   ```python
   from pyspark.sql import SparkSession

   spark = SparkSession.builder \
       .appName("PySpark Shell") \
       .getOrCreate()

   print(spark.version)
   ```

## Step 5: Authentication and Accessing Spark Schemas

1. **Configure Authentication**:
   ```sh
   export SPARK_OPTS="--conf spark.authenticate=true --conf spark.authenticate.secret=your_secret"
   ```

2. **Access Spark Schemas**:
   - Use PySpark to query and access Spark schemas.
   ```python
   df = spark.sql("SELECT * FROM your_schema.your_table")
   df.show()
   ```

## Complete Script

Here’s a complete script that you can run inside the Docker shell to achieve the desired updates:

```sh
#!/bin/sh

# Step 1: Install Miniconda and Update Python
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init
source ~/.bashrc
conda create --name myenv python=3.9 -y
conda activate myenv

# Step 2: Install and Update Spark
conda install pyspark=3.2.0 -y

# Step 3: Configure Jupyter Kernel
conda install jupyter -y
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

# Step 4: Configure PySpark in Jupyter
export SPARK_HOME=~/miniconda3/envs/myenv/lib/python3.9/site-packages/pyspark
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH

# Step 5: Authentication and Accessing Spark Schemas
export SPARK_OPTS="--conf spark.authenticate=true --conf spark.authenticate.secret=your_secret"

# Start Jupyter Notebook
jupyter notebook
```

## Instructions to Use the Script

1. Save the script to a file named `update_and_configure.sh`.
2. Make the script executable:
   ```sh
   chmod +x update_and_configure.sh
   ```
3. Run the script:
   ```sh
   ./update_and_configure.sh
   ```

This script will update Python using Miniconda, install and update Spark, configure a Jupyter kernel, and ensure that PySpark is correctly set up. You can then use Jupyter notebooks to access Spark schemas with the updated environment.