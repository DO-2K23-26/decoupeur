from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for f in features:
            self.downs.append(ConvBlock(in_channels, f))
            in_channels = f
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(f * 2, f))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # upsample
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)  # conv block
        
        return torch.sigmoid(self.final_conv(x))


def segment_person(model, frame_bgr, img_size=512):
    """Segment a person from a frame in real-time."""
    h, w = frame_bgr.shape[:2]
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(rgb, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.tensor(img_normalized.transpose(2, 0, 1), 
                              dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()
    
    # Resize to original size
    mask = cv2.resize(pred, (w, h))
    
    # Post-processing
    mask = (mask > 0.4).astype(np.uint8) * 255
    
    # Keep only largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Soft edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply mask to frame (keep person, replace background with black)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch > 127, frame_bgr, 0)
    
    return result.astype(np.uint8), mask


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create and load model
print('Loading model...')
model = UNet(in_channels=3, out_channels=1).to(device)
checkpoint = torch.load('epoch-39.pth', map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()
print('Model loaded!')

# Start real-time camera segmentation
print('Starting real-time segmentation. Press Q to quit.')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time

frame_count = 0
fps_clock = cv2.getTickCount()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Segment
        segmented, mask = segment_person(model, frame)
        
        # Display results side by side
        h, w = frame.shape[:2]
        combined = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), segmented])
        
        # FPS counter
        frame_count += 1
        if frame_count % 10 == 0:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_clock) * 10
            fps_clock = cv2.getTickCount()
            cv2.putText(combined, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Original | Mask | Segmented', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print('Segmentation terminated.')