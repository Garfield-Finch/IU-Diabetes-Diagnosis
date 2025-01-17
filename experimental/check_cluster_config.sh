import subprocess
import os

def run_command(cmd):
    print(f"\nExecuting: {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output.decode()}")

def check_environment():
    """Check cluster environment configuration"""
    # Check Java
    run_command("java -version")
    
    # Check Hadoop config
    print("\nHadoop Configuration Files:")
    hadoop_conf = "/usr/local/hadoop/etc/hadoop"
    files_to_check = [
        "core-site.xml",
        "hdfs-site.xml",
        "yarn-site.xml",
        "mapred-site.xml"
    ]
    
    for file in files_to_check:
        path = os.path.join(hadoop_conf, file)
        if os.path.exists(path):
            print(f"\nContents of {file}:")
            run_command(f"cat {path}")
    
    # Check network configuration
    print("\nNetwork Configuration:")
    run_command("hostname -f")
    run_command("ip addr")
    
    # Check Hadoop/YARN status
    print("\nHadoop/YARN Status:")
    run_command("jps")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    vars_to_check = [
        "JAVA_HOME",
        "HADOOP_HOME",
        "HADOOP_CONF_DIR",
        "YARN_CONF_DIR",
        "SPARK_HOME",
        "PYTHONPATH",
        "PYSPARK_PYTHON",
        "PYSPARK_DRIVER_PYTHON"
    ]
    
    for var in vars_to_check:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

if __name__ == "__main__":
    check_environment()
