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

## Methodology

We used the following method to analyze issues of DNN libraries and create corresponding reproduction scripts. For each DNN library:

- Access the issues from the corresponding GitHub repository, e.g., [https://github.com/tensorflow/tensorflow/issues](https://github.com/tensorflow/tensorflow/issues).
- Filter the set of issues based on the following criteria:
	- Issues within a specific time range;
	- Issues that are closed;
	- Issues classified as a bug;
	- Issues that have pull requests (more likely to be a library misuse, otherwise).
<!--- Go to the issues that has a pull request associated with it to avoid bug reports that are merely misuses of the library and instead of a fix, the solution is a workaround (hence no pull requests).
-->
- Determine whether the bug is deep-learning specific (e.g., related to the construction or usage of models) or not (e.g. related to math operations, data structures etc.). This data set focuses on deep-learning specific bugs.
- Create a reproduction script:
	- 	Create a directory named with the identifier of the GitHub issue 
	-  Add a self-contained bug reproduction script, ```reproduce_bugs.sh```, in that directory. The script will (1) create an execution environment and (2) run one single pytest test on that environment.
	- 	Prepare the execution environment: If the bug is dependent on CUDA, create a Docker container for isolation. Otherwise, create a conda virtual environment.
	-  Create a test that is faithful to the one described in the issue. The test run shold report the following outcomes: 
		-  0 - reproduction successful 
		-  1 - reproduction failed (e.g., this should not happen)
		-  2 - assumption broken (e.g., expecting to run the test on linux but attempting to run on macos)<!--	Try to reproduce the bug using a conda enviornment if there is no CUDA dependency. For 
-->
- Add the details of the bug in the spreadsheets of each library (e.g. ```bug_dataset_jax.csv```)

## Known Issues
- Packages outside conda environments are being accessed and it says "requirement already satisfied" while referencing the package installed outside the conda environment:
	- Solution: [StackOverflow](https://stackoverflow.com/questions/59044844/local-pip-in-conda-environment-checks-globally-and-says-requirement-already-sati)
	  ```Shell
	  pip freeze --user > packages.txt
	  pip uninstall -r packages.txt
   	  ```
