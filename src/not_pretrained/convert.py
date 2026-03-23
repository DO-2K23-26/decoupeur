import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import sys

# Configuration
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 512
OPACITY = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model
print('Loading model...')
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=3, classes=1, activation=None)

checkpoint = torch.load('unet_segmentation+resnet.pth', map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print('Model loaded!')


def add_antennas(image, mask):
    """Add green antennas on top of the detected head."""
    h, w = mask.shape[:2]
    
    # Get contours to find the head boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # Get the largest contour (the person)
    contour = max(contours, key=cv2.contourArea)
    
    # Find the topmost point in the contour
    topmost_idx = contour[:, 0, 1].argmin()
    top_point = contour[topmost_idx][0]
    top_x, top_y = int(top_point[0]), int(top_point[1])
    
    # Find points around the top to determine head width
    # Look for contour points near the top
    nearby_points = []
    for i in range(max(0, topmost_idx - 20), min(len(contour), topmost_idx + 20)):
        pt = contour[i][0]
        if abs(pt[1] - top_y) < 30:  # Points within 30 pixels vertically
            nearby_points.append(pt)
    
    # Calculate head width from nearby points
    if len(nearby_points) > 2:
        head_width = (max(p[0] for p in nearby_points) - min(p[0] for p in nearby_points))
    else:
        head_width = 50
    
    # Antenna parameters
    antenna_length = int(head_width * 0.8)
    antenna_color = (0, 255, 0)  # Green in BGR
    antenna_thickness = 4
    
    # Antenna bases (positioned on head width - more spread)
    left_base_x = top_x - int(head_width * 0.5)
    right_base_x = top_x + int(head_width * 0.5)
    base_y = top_y + int(head_width * 0.25)  # Positioned lower on the head
    
    # Antenna endpoints (angled outward and upward)
    left_end_x = left_base_x - int(antenna_length * 0.4)
    left_end_y = top_y - antenna_length
    
    right_end_x = right_base_x + int(antenna_length * 0.4)
    right_end_y = top_y - antenna_length
    
    # Draw curved antennas using Bezier curve
    antenna_color_bgr = antenna_color
    
    # Left antenna
    control_points_left = [
        (left_base_x, base_y),
        (left_base_x - int(antenna_length * 0.3), top_y - int(antenna_length * 0.6)),
        (left_end_x, left_end_y)
    ]
    
    # Right antenna
    control_points_right = [
        (right_base_x, base_y),
        (right_base_x + int(antenna_length * 0.3), top_y - int(antenna_length * 0.6)),
        (right_end_x, right_end_y)
    ]
    
    # Draw curved antennas with Bezier interpolation
    prev_left_pt = None
    prev_right_pt = None
    
    for i in range(1, 101):
        t = i / 100.0
        # Quadratic Bezier curve formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        
        # Left antenna
        left_pt = np.array([
            (1-t)**2 * control_points_left[0][0] + 2*(1-t)*t * control_points_left[1][0] + t**2 * control_points_left[2][0],
            (1-t)**2 * control_points_left[0][1] + 2*(1-t)*t * control_points_left[1][1] + t**2 * control_points_left[2][1]
        ]).astype(int)
        
        # Right antenna
        right_pt = np.array([
            (1-t)**2 * control_points_right[0][0] + 2*(1-t)*t * control_points_right[1][0] + t**2 * control_points_right[2][0],
            (1-t)**2 * control_points_right[0][1] + 2*(1-t)*t * control_points_right[1][1] + t**2 * control_points_right[2][1]
        ]).astype(int)
        
        if prev_left_pt is not None:
            cv2.line(image, prev_left_pt, tuple(left_pt), antenna_color_bgr, antenna_thickness)
            cv2.line(image, prev_right_pt, tuple(right_pt), antenna_color_bgr, antenna_thickness)
        
        prev_left_pt = tuple(left_pt)
        prev_right_pt = tuple(right_pt)
    
    # Add spheres at the end of antennas
    sphere_radius = 8
    cv2.circle(image, (left_end_x, left_end_y), sphere_radius, antenna_color_bgr, -1)
    cv2.circle(image, (right_end_x, right_end_y), sphere_radius, antenna_color_bgr, -1)
    
    return image



def extract_and_colorize(image_path, output_path=None):
    """Extract mask and colorize person in green with 0.3 opacity."""
    # Load image
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    h, w = frame_bgr.shape[:2]
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = (img_resized / 255.0 - MEAN) / STD
    
    # Convert to tensor
    img_tensor = torch.tensor(img_normalized.transpose(2, 0, 1), 
                              dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    print('Processing image...')
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0, 0].cpu().numpy()
    
    # Resize to original size
    mask = cv2.resize(pred, (w, h))
    
    # ── POST-PROCESSING ──
    # Threshold à 0.4
    mask = (mask > 0.4).astype(np.uint8) * 255
    
    # Garde uniquement le plus grand contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    
    # Fermeture morphologique avec kernel 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Bords doux
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # ── FIN POST-PROCESSING ──
    
    # Create green overlay (BGR format: 0, 255, 0 for green)
    green_overlay = np.zeros_like(frame_bgr)
    green_overlay[:, :] = (0, 255, 0)  # Green color in BGR
    
    # Convert mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    # Apply opacity (0.3) to green overlay
    green_overlay = green_overlay.astype(np.float32)
    result = frame_bgr.astype(np.float32)
    result = result * (1 - mask_3ch * OPACITY) + green_overlay * (mask_3ch * OPACITY)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Add antennas on top of head
    result = add_antennas(result, mask)
    
    # Save result
    if output_path is None:
        output_path = image_path.replace('.', '_colored.')
    
    cv2.imwrite(output_path, result)
    print(f'Result saved to: {output_path}')
    
    # Also save the mask
    mask_path = image_path.replace('.', '_mask.')
    cv2.imwrite(mask_path, mask)
    print(f'Mask saved to: {mask_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_and_colorize(image_path, output_path)
