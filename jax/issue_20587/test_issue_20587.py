import jax
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cephes4py
import pytest

def test_f():
    issue_no = '20587'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()

    b = 14.25
    n = b - 1
    k = jnp.arange(40)

    scipy_output = scipy.special.beta(1 + n - k, 1 + k)
    jax_output = jax.scipy.special.beta(1 + n - k, 1 + k)
    cephes_output = np.array([cephes4py.beta(1 + n - k_, 1 + k_) for k_ in k])

    print("scipy_output:", scipy_output)
    print("jax_output:", jax_output)
    print("cephes_output:", cephes_output)

    plt.plot(k, scipy_output, label="scipy", ls=":")
    plt.plot(k, jax_output, label="jax")
    plt.plot(k, cephes_output, label="cephes", ls="-.")
    plt.legend()
    
    save_file_path = '/tmp/dnnbugs_pytorch_issue_20587_plot.png'
    plt.savefig(save_file_path)
    plt.close()
    print(f"Plot saved to: {save_file_path}")

    with pytest.raises(AssertionError) as e_info:
        np.testing.assert_allclose(scipy_output, jax_output, rtol=1e-4, atol=1e-6) # jax.scipy.special.beta results are different from scipy.special.beta
    print(f'{e_info.type.__name__}: {e_info.value}')
