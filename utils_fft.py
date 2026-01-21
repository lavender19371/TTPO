import torch

def gaussian_lowpass_filter(shape, D0=0.1, device="cuda"):
    H, W = shape
    y = torch.linspace(-0.5, 0.5, H, device=device)
    x = torch.linspace(-0.5, 0.5, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    
    D = torch.sqrt(X**2 + Y**2)  # max distance sqrt(0.5² + 0.5²) ≈ 0.707
    
    G = torch.exp(- (D**2) / (2 * ((D0*0.707)**2)))
    return G

def apply_highpass(image_tensor, D0=0.1, filter=None):
    H, W = image_tensor.shape[-2:]
    if filter is None:
        G = gaussian_lowpass_filter((H, W), D0=D0, device=image_tensor.device)
    else:
        G = filter
    
    fft = torch.fft.fftshift(torch.fft.fft2(image_tensor))
    filtered = fft * (1-G)
    return torch.fft.ifft2(torch.fft.ifftshift(filtered)).real

def apply_lowpass(image_tensor, D0=0.1, filter=None):
    H, W = image_tensor.shape[-2:]
    if filter is None:
        G = gaussian_lowpass_filter((H, W), D0=D0, device=image_tensor.device)
    else:
        G = filter
    
    fft = torch.fft.fftshift(torch.fft.fft2(image_tensor))
    filtered = fft * G
    return torch.fft.ifft2(torch.fft.ifftshift(filtered)).real