import jax

def test_f():
    issue_no = '18218'
    print('Jax issue no.', issue_no)
    jax.print_environment_info()
    x = jax.tree_util.tree_structure(0)
    y = jax.tree_util.tree_structure((0,0))
    assert x.compose(y).num_leaves == 4 # Correct behavior should be 2