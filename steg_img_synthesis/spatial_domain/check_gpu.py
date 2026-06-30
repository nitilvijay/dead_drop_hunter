import torch
import intel_extension_for_pytorch as ipex

def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if XPU (Intel GPU) is available
    is_xpu_available = torch.xpu.is_available()
    print(f"Intel GPU (XPU) Available: {is_xpu_available}")
    
    if is_xpu_available:
        device_count = torch.xpu.device_count()
        print(f"Number of XPU Devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.xpu.get_device_name(i)
            print(f"Device {i} Name: {device_name}")
            
        # Create a simple tensor and move it to XPU
        device = torch.device("xpu")
        x = torch.randn(5, 5).to(device)
        y = torch.randn(5, 5).to(device)
        z = x + y
        
        print("\nTensor operation on XPU successful!")
        print(f"Result device: {z.device}")
    else:
        print("\nIntel GPU not detected by PyTorch/IPEX.")
        print("Please ensure the Intel GPU drivers and oneAPI Base Toolkit are installed.")

if __name__ == "__main__":
    check_gpu()
