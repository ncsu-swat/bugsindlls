# dnnbugs
A benchmark of reproducible bugs in DNN libraries

## Prerequisites
- [Docker](https://docs.docker.com/engine/install/)
- Conda ([Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/))
- [Only if you have an NVIDIA GPU] [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 


## Steps to configure
1. Clone this repository: ```$> git clone https://github.com/ncsu-swat/dnnbugs.git```
2. Add dnnbugs to your PATH: ```$> export PATH=$PATH:<dnnbugs_path>/framework```


## Commands

| Command  | Description |
| -------- | ------- |
| list-tests  | List the tests available on this dataset |
| run-test | Runs one test |
| run-tests | Runs several tests |
| show-info | Shows information about the tests available on this benchmark |
| stats | Shows statistics about this dataset (e.g., number of tests that require GPU, number of tests that reproduce bugs in C or Python code, etc.) |


<!---
>>>>>>> f880b48 (organizing framework)
## How to reproduce

- Change the current directory to the specific bug's directory. For example:

```Shell
cd jax/issue_18218
```

- Execute the script "reproduce_bug.sh"

```Shell
./reproduce_bug.sh
```

- Upon successful reproduction, the test should pass. Look for "1
  passed in" towards the end of the output.
  
-->

## Example usage

- Show all reproducible bugs (tests) on this dataset: 

```Shell
$> list-tests
```

- Help to use command show-info: 

```Shell
$> show-info --help
```

- Show information about bug 120903 from pytorch (id obtained from command above): 

```Shell
$> show-info --library-name pytorch --bug-id 120903
```

- Reproduce that bug:

```Shell
$> run-test --library-name pytorch --bug-id 120903
```

- Reproduce tests, saving logs in the provided directory:

```Shell
$> run-tests --log-directory ~/dnn-logs
```

- Show statistics about the dataset with the buggy files in a trie format

```Shell
$> stats --print-format trie
```

# Tool integration

Testing tools can be integrated to the dataset to recreate the environment of a specific bug and execute the tool on that environment to see if it can be reproduced with the tool. To integrate, there are two requirements:

1. A docker container with the environment of the tool
2. A script that takes the name of the library as an argument and that can trigger the execution of the tool for that library

Once the docker container is built and is running, providing the name of the container and the location of the script that can execute the tool (either absolute path or relative to the root of the repository) to the command `run-tool` in the framework along with the library and bug id will first reproduce the bug seperately and then update the environment inside the docker to run the tool with the script provided.

## Demonstration

A demonstration of this integration is provided with FreeFuzz, a state-of-the-art testing tool. To install and run a container for freefuzz:

```Shell
$> cd tool-integration/FreeFuzz && bash install_freefuzz_docker.sh
```

To run the tool in a bug's environment (e.g. pytorch issue_122016)

```Shell
$> run-tool --container freefuzz --library-name pytorch --bug-id 122016 --run-script tool-integration/FreeFuzz/run_freefuzz_docker.sh
```

Note: For demonstration purposes, the `run_freefuzz_docker.sh` script uses a demo configuration of freefuzz. For a full run, this script would need to be updated with a different config file. For more instructions, see the [documentation](https://github.com/ise-uiuc/FreeFuzz/blob/main/README.md) of FreeFuzz.