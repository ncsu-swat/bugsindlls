import os
import time
import threading
import subprocess
import jax
import jax.numpy as jnp
import pytest

def f(stop_event):
    matrix_shape = (1000, 1000)
    while not stop_event.is_set():
        for i in range(20_000):
            if stop_event.is_set():
                break
            key = jax.random.PRNGKey(i)
            a_key, b_key = jax.random.split(key, 2)
            a = jax.random.normal(a_key, matrix_shape)
            b = jax.random.normal(b_key, matrix_shape)
            c = a @ b

def run_test(env, value, duration):
    os.environ[env] = value
    print(f"\nTest with {env}={value}")

    stop_event = threading.Event()
    jax_test_thread = threading.Thread(target=f, args=(stop_event,))
    jax_test_thread.start()

    time.sleep(duration)

    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout

    stop_event.set()
    jax_test_thread.join()

    print(nvidia_smi_output)


def test_f():
    issue_no = '19035'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    run_test('XLA_PYTHON_CLIENT_PREALLOCATE','false', 5)
    run_test('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.99', 5)
