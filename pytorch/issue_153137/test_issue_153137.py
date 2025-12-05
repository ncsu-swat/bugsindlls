import torch
import pytest

def test_f():
    
    print(f"PyTorch Version: {torch.__version__}")

    # Common parameters for torch.batch_norm
    weight_param = None
    bias_param = None
    is_training_param = True # Error occurs with True or False
    momentum_param = 0.1
    eps_param = 1e-5
    cudnn_enabled_param = True # Also occurs with False on GPU

    # --- Scenario 1: running_mean is Tensor, running_var is None ---
    print("\n--- Scenario 1: running_mean is Tensor, running_var is None ---")
    # Input tensor
    input_tensor_shape = (3, 4, 5) # N, C, D*
    num_features = input_tensor_shape[1]

    # CPU
    print("  CPU (Scenario 1):")
    try:
        input_tensor_cpu = torch.randn(input_tensor_shape)
        running_mean_param_cpu = torch.randn(num_features)
        running_var_param_cpu = None
        
        torch.batch_norm(
            input_tensor_cpu,
            weight_param,
            bias_param,
            running_mean_param_cpu,
            running_var_param_cpu,
            is_training_param,
            momentum_param,
            eps_param,
            cudnn_enabled_param
        )
        print("    CPU: Error not triggered.")
        assert True
    except RuntimeError as e:
        print(f"    CPU Error: {e}")
        if "Expected has_running_mean == has_running_var to be true, but got false" in str(e):
            print("    CPU: Successfully triggered the target error (unexpected based on current behavior).")

    # GPU
    if torch.cuda.is_available():
        print("  GPU (Scenario 1):")
        with pytest.raises(RuntimeError) as e_info:
            input_tensor_gpu = torch.randn(input_tensor_shape).cuda()
            running_mean_param_gpu = torch.randn(num_features).cuda()
            running_var_param_gpu = None
            
            torch.batch_norm(
                input_tensor_gpu,
                weight_param,
                bias_param,
                running_mean_param_gpu,
                running_var_param_gpu,
                is_training_param,
                momentum_param,
                eps_param,
                cudnn_enabled_param
            )
            print("    GPU: Error not triggered (unexpected for this specific error message).")
        print(f'{e_info.type.__name__}: {e_info.value}')
    else:
        print("  GPU (Scenario 1): CUDA not available, skipping GPU test.")

    # --- Scenario 2: running_mean is None, running_var is Tensor ---
    print("\n--- Scenario 2: running_mean is None, running_var is Tensor ---")

    # CPU
    print("  CPU (Scenario 2):")
    try:
        input_tensor_cpu = torch.randn(input_tensor_shape)
        running_mean_param_cpu = None
        running_var_param_cpu = torch.randn(num_features)
        
        torch.batch_norm(
            input_tensor_cpu,
            weight_param,
            bias_param,
            running_mean_param_cpu,
            running_var_param_cpu,
            is_training_param,
            momentum_param,
            eps_param,
            cudnn_enabled_param
        )
        print("    CPU: Error not triggered.")
        assert True
    except RuntimeError as e:
        print(f"    CPU Error: {e}")
        if "Expected has_running_mean == has_running_var to be true, but got false" in str(e):
            print("    CPU: Successfully triggered the target error (unexpected based on current behavior).")

    # GPU
    if torch.cuda.is_available():
        print("  GPU (Scenario 2):")
        with pytest.raises(RuntimeError) as e_info:
            input_tensor_gpu = torch.randn(input_tensor_shape).cuda()
            running_mean_param_gpu = None
            running_var_param_gpu = torch.randn(num_features).cuda()
            
            torch.batch_norm(
                input_tensor_gpu,
                weight_param,
                bias_param,
                running_mean_param_gpu,
                running_var_param_gpu,
                is_training_param,
                momentum_param,
                eps_param,
                cudnn_enabled_param
            )
            print("    GPU: Error not triggered (unexpected for this specific error message).")
        print(f'{e_info.type.__name__}: {e_info.value}')
    else:
        print("  GPU (Scenario 2): CUDA not available, skipping GPU test.")
