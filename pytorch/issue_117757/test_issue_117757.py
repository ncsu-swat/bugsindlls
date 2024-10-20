import unittest
import torch
import operator
import pytest


class TestUnaryFunctions(unittest.TestCase):
    def test_unary_functions(self):
        issue_no = '120387'
        print('Pytorch issue no.', issue_no)
        for op in (operator.abs, ):
                print("iteration %s" % op)
                def fn(x, y):
                    return x * op(y)

                arg = torch.ones(4)*4
                opt_fn = torch._dynamo.optimize(nopython=True, backend="inductor", dynamic=True)(fn)
                print("tensor*abs(constant)")

                # This fail on second iteration.
                # tensor([-2., -2., -2., -2.]) VS tensor([2., 2., 2., 2.])
                # print(opt_fn(arg, -2), fn(arg, -2))

                self.assertFalse(torch.equal(opt_fn(arg, -2), fn(arg, -2)), "Tensors are not equal")