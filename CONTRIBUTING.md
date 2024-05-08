# Contributing to dnnbugs

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)


## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

<!--We used the following method to analyze issues of DNN libraries and create corresponding reproduction scripts. For each DNN library:-->
<!--- Go to the issues that has a pull request associated with it to avoid bug reports that are merely misuses of the library and instead of a fix, the solution is a workaround (hence no pull requests).
-->
- Identify issues to analyze from from the GitHub repository of the library of interest, e.g., [tensorflow](https://github.com/tensorflow/tensorflow/issues). You need to filter issues based on the following criteria:
   - Issues that are closed;
	- Issues classified as a bug;
	- Issues that have pull requests or commits associated (so that we can identify the code fixes).
	- Issues within a specific time range (favor more recent bugs);
<!--- Determine whether the bug is deep-learning specific (e.g., related to the construction or usage of models) or not (e.g. related to math operations, data structures etc.). This data set focuses on deep-learning specific bugs.-->
- Create a reproduction script:
	- 	Create a directory named with the identifier of the GitHub issue 
	-  Add a self-contained bug reproduction script, ```reproduce_bugs.sh```, in that directory. The script will (1) create an execution environment and (2) run one single pytest test on that environment.
	- 	Prepare the execution environment: If the bug is dependent on CUDA, create a Docker container for isolation. Otherwise, create a conda virtual environment.
	-  Create a test that is faithful to the one described in the issue. The test run shold report the following outcomes: 
		-  0 - reproduction successful 
		-  1 - reproduction failed (e.g., this should not happen)
		-  2 - assumption broken (e.g., expecting to run the test on linux but attempting to run on macos)
- Add the details of the bug in the spreadsheets of each library (e.g. ```bug_dataset_jax.csv```)

## Reporting Bugs


#### Before Submitting a Bug Report

A good bug report should be self contained; it should not leave others needing to chase you up for more information. We ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions. Make sure that you followed the instructions in [README.md](https://github.com/ncsu-swat/dnnbugs/README.md). 
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/ncsu-swat/dnnbugs/issues?q=label%3Abug).

#### Preparing a Bug Report

- Collect information about the problem you experienced.
	- If you failed to reproduce a bug in our dataset, please share the execution logs with us. Example command to collect log (log.txt) for bug <id> in library <library>:<br>```$> run-test -l <library> -i <id> 2>&1 | tee log.txt```
- File an [issue](https://github.com/ncsu-swat/dnnbugs/issues) describing the steps to reproduce the problem.

For reference, [here](https://github.com/ncsu-swat/dnnbugs/issues/23) is an example of a bug report.


<!---

## Known Issues
- Packages outside conda environments are being accessed and it says "requirement already satisfied" while referencing the package installed outside the conda environment:
        - Solution: [StackOverflow](https://stackoverflow.com/questions/59044844/local-pip-in-conda-environment-checks-globally-and-says-requirement-already-sati)
          ```Shell
          pip freeze --user > packages.txt
          pip uninstall -r packages.txt
          ```
          
-->          
