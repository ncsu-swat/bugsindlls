import unittest
import torch

class TestUnaryFunctions:
    def test_unary_functions(self):
        for op in (operator.abs, ):
            print("iteration %s" % op)
            
            def fn(x, y):
                return x * op(y)

            arg = torch.ones(4) * 4
            opt_fn = torch._dynamo.optimize(nopython=True, backend="inductor", dynamic=True)(fn)
            
            print("tensor*abs(constant)")
            
            # Uncomment the print to see the output
            # print(opt_fn(arg, -2), fn(arg, -2))
            
            # Assert equality to check if the optimization works correctly
            assert opt_fn(arg, -2) == fn(arg, -2), "Test failed!"

if __name__ == "__main__":
    test = TestUnaryFunctions()
    test.test_unary_functions()

