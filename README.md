# dnnbugs
A benchmark of reproducible bugs in DNN libraries

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
- Add the details of the bug in the spreadsheet ```bug_dataset.xlsx```

