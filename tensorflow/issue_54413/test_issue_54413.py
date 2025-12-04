import tensorflow as tf
import pytest

def test_f():
    bytes_or_text = "hello"
    encoding = "valid"
    t1 = tf.compat.as_text(bytes_or_text, encoding=encoding)
    print(t1) # hello
    assert t1 == "hello"
    with pytest.raises(LookupError) as e_info:
        t2 = tf.compat.as_bytes(bytes_or_text,encoding=encoding)
        # LookupError: unknown encoding: valid
    print(f'{e_info.type.__name__}: {e_info.value}')
