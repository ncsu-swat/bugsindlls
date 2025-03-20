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
| run-tool | Runs a testing tool in the environment of a bug in the dataset to assess the tool's ability to reproduce the bug |


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

Sample output:

```Shell
...
Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] torch==2.2.0+cpu
[conda] torch                     2.2.0+cpu                pypi_0    pypi
====== test session starts ======
platform linux -- Python 3.10.0, pytest-8.2.0, pluggy-1.5.0
rootdir: /home/mnaziri/Documents/DL_Testing/dnnbugs/pytorch/issue_120903
collected 1 item                                                                            

test_issue_120903.py Pytorch issue no. 120903
Seed:  120903
RuntimeError: !needs_dynamic_casting<func_t>::check(iter) INTERNAL ASSERT FAILED at "../aten/src/ATen/native/cpu/Loops.h":310, please report a bug to PyTorch. 
====== 1 passed in 0.96s ======
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

To run the tool in a bug's environment (e.g. pytorch issue_117033)

```Shell
$> run-tool --container freefuzz --library-name pytorch --bug-id 117033 --run-script tool-integration/FreeFuzz/run_freefuzz_docker.sh
```

Alternatively, to run the tool by specifying a library version instead of a bug ID (e.g., torch version 2.3.1):

```Shell
$> run-tool --container freefuzz --library-name pytorch --use-library-version 2.3.1 --run-script tool-integration/FreeFuzz/run_freefuzz_docker.sh
```

Sample output:

```Shell
...
====== test session starts ======
platform linux -- Python 3.10.0, pytest-8.3.3, pluggy-1.5.0
...
IndexError: list index out of range
====== 1 passed in 1.09s ======
...
Updating environment in the container of the testing tool
Successfully copied 16.4kB to freefuzz:/tmp/issue_117033
__pycache__  install_environment.sh  reproduce_bug.sh  requirements.txt  test_issue_117033.py
...
Running the testing tool on the environment of the bug
Testing on  ['torch']
torch.log2
...
torch.nn.MaxUnpool2d
../aten/src/ATen/native/cuda/MaxUnpooling.cu:47: max_unpooling2d_forward_kernel: block: [0,0,0], thread: [1,0,0] Assertion `maxind >= 0 && maxind < outputImageSize` failed.
../aten/src/ATen/native/cuda/MaxUnpooling.cu:47: max_unpooling2d_forward_kernel: block: [0,0,0], thread: [2,0,0] Assertion `maxind >= 0 && maxind < outputImageSize` failed.
...

```

After the execution completes, the output can be checked for all failures produced by FreeFuzz (saved inside the container) and manually inspected to see if any failure that can reveal the bug in question was generated by the tool. In this case, the failure generated by FreeFuzz could not generate a failure that can reveal this bug since the API "torch._dynamo.export" is not supported by FreeFuzz.

Note: For demonstration purposes, the `run_freefuzz_docker.sh` script uses a demo configuration of freefuzz. For a full run, this script would need to be updated with a different config file. For more instructions, see the [documentation](https://github.com/ise-uiuc/FreeFuzz/blob/main/README.md) of FreeFuzz.