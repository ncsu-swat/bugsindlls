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

- Upon successful reproduction, the test will pass

## Methodology

- Go to the issues tab in the GitHub repository of a library (e.g. tensorflow).
- Apply a filter which has a specific date range, closed issues, label type bug (if present) and a linked pull request (to avoid bug reports which are merely misuses of the library and instead of a fix, the solution is a workaround). The exact filters are provided in the spreadsheet (bug_dataset.xlsx)
- Assess the bug to determine whether the bug is deep learning specific or not (e.g. math operations, data structures etc.) and only considering deep learning specific bugs (bugs that have an association to model creations and usages)
- Try to reproduce the bug using a conda enviornment if there is no CUDA dependency. For CUDA dependent cases, use docker.
- If successful, create a directory with the issue name and add a self contained bug reproduction script (reproduce_bugs.sh) in the directory that uses python scripts containing the buggy code to reproduce the bug and return exit codes depending on the status (0 - reproduction succesful, 1 - reproduction failed, 2 - assumption broken)
- Add the details of the bug in the spreadsheet (bug_dataset.xlsx)
