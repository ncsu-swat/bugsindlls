import torch
import operator
import unittest
import pytest

class TestUnaryFunctions(unittest.TestCase):
    def test_unary_functions(self):
        issue_no = '118231'
        print('Pytorch issue no.', issue_no)

        for op in (operator.pos,):
            with self.subTest(op=op):
                def fn(x, y):
                    return x * op(y)

                opt_fn = torch.compile(fullgraph=True, dynamic=True)(fn)
                tensor1 = torch.ones(4)
                tensor2 = torch.ones(4)

                def test(arg1, arg2):
                    self.assertEqual(opt_fn(arg1, arg2), fn(arg1, arg2))
                
                with pytest.raises(torch._dynamo.exc.Unsupported) as e_info:
                    test(-2, -2)
                print(f'{e_info.type.__name__}: {e_info.value}')
