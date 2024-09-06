import torch
from transformers.utils import logging as hf_logging
import logging
import pytest

hf_logger = hf_logging.get_logger(__name__)
logger = logging.getLogger(__name__)

def f_print(x):
    print("abc")
    return x + 1

def f_logger(x):
    logger.log("abc")
    return x + 1

def f_hf_logger(x):
    hf_logger.warning_once("abc")  # HF's logger
    return x + 1

def test_f():
    issue_no = '123288'
    print('Pytorch issue no.', issue_no)

    print("Dynamo full graph export with print:")
    with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:
        torch._dynamo.export(f_print, torch.randn(2, 3))
    print(f'{e_info.type.__name__}: {e_info.value}')

    print("Dynamo full graph export with logger:")
    with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:
        torch._dynamo.export(f_logger, torch.randn(2, 3))
    print(f'{e_info.type.__name__}: {e_info.value}')

    print("Dynamo full graph export with HF's logger:")
    with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:
        torch._dynamo.export(f_hf_logger, torch.randn(2, 3))
    print(f'{e_info.type.__name__}: {e_info.value}')
