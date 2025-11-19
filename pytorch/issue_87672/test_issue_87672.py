import pytest
import torch

def test_f():
    
    input_ = [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ]
    ]

    input_ = torch.tensor(input_, dtype=torch.float32)
    with pytest.raises(RuntimeError) as e_info:
        input_.set_(torch.tensor(
            [
                [
                    [21, 22, 23],
                    [24, 25, 26],
                    [27, 28, 29]
                ],
                [
                    [31, 32, 33],
                    [34, 35, 36],
                    [37, 38, 39]
                ]
            ]))

        print(input_)
    print(f'{e_info.type.__name__}: {e_info.value}')
