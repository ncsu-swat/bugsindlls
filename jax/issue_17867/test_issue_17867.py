import subprocess
import os
import sys
import jax
import pytest


def f():
    test_functions = [
        "random_test.py::LaxRandomTest::test_copy0",
        "random_test.py::LaxRandomWithRBGPRNGTest::test_copy0",
        "random_test.py::LaxRandomWithUnsafeRBGPRNGTest::test_copy0"
    ]

    results = []
    
    for test_function in test_functions:

        command = f"cd jax/tests/; pytest {test_function}"
        print(f"Running command: {command}")
        # Get the output of the command
        std_out, std_err = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

        results.append(std_out.decode('utf-8'))
    return results

def test_f():
    issue_no = '17867'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    results = f()

    for result in results:
        assert "RuntimeWarning: divide by zero" in result