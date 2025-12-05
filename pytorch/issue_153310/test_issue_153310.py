import pytest
import torch

def test_f():
    
    print(torch.__version__)
    cpu_res=torch.tensor([0., 0., 0., 0., 0., 0.])
    # self_cpu = list(torch.jit.load("self_cpu.pt").parameters())[0]
    self_cpu = torch.tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,
                0.0000e+00,  0.0000e+00], dtype=torch.float32)
    observer_on_cpu = torch.tensor(False)
    fake_quant_on_cpu = torch.tensor(False)
    running_min_cpu = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    running_max_cpu = torch.tensor([0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
            0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010])
    scale_cpu = torch.tensor([0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
            0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010])
    zero_point_cpu = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32)

    try:
        res=torch.fused_moving_avg_obs_fake_quant(
            self_cpu,
            observer_on_cpu,
            fake_quant_on_cpu,
            running_min_cpu,
            running_max_cpu,
            scale_cpu,
            zero_point_cpu,
            0,
            0,
            1,
            0,
            False,
            False
        )
        print(res)
        assert torch.allclose(res, cpu_res)
    except RuntimeError as e:
        print("CPU Error:", e)

    # run on gpu
    self_cpu_cuda = self_cpu.cuda()
    observer_on_cpu_cuda = observer_on_cpu.cuda()
    fake_quant_on_cpu_cuda = fake_quant_on_cpu.cuda()
    running_min_cpu_cuda = running_min_cpu.cuda()
    running_max_cpu_cuda = running_max_cpu.cuda()
    scale_cpu_cuda = scale_cpu.cuda()
    zero_point_cpu_cuda = zero_point_cpu.cuda()

    with pytest.raises(RuntimeError) as e_info:
        torch.fused_moving_avg_obs_fake_quant(
            self_cpu_cuda,
            observer_on_cpu_cuda,
            fake_quant_on_cpu_cuda,
            running_min_cpu_cuda,
            running_max_cpu_cuda,
            scale_cpu_cuda,
            zero_point_cpu_cuda,
            0,
            0,
            1,
            0,
            False,
            False
        )
        print("GPU Success")
    print(f'{e_info.type.__name__}: {e_info.value}')
