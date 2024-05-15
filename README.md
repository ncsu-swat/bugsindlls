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
