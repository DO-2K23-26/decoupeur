import socket
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import threading
import time

# Configuration
HOST = '127.0.0.1'
PORT = 5001
BUFFER_SIZE = 65536

# Model configuration
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMG_SIZE = 512
OPACITY = 0.3  # Green overlay opacity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Server device: {device}')

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


def extract_mask(frame_bgr):
    """Extract the segmentation mask from a frame."""
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
    
    return mask


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
    left_end_y = base_y - antenna_length
    
    right_end_x = right_base_x + int(antenna_length * 0.4)
    right_end_y = base_y - antenna_length
    
    # Draw curved antennas using Bezier curve
    antenna_color_bgr = antenna_color
    
    # Left antenna
    control_points_left = [
        (left_base_x, base_y),
        (left_base_x - int(antenna_length * 0.3), base_y - int(antenna_length * 0.6)),
        (left_end_x, left_end_y)
    ]
    
    # Right antenna
    control_points_right = [
        (right_base_x, base_y),
        (right_base_x + int(antenna_length * 0.3), base_y - int(antenna_length * 0.6)),
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


def make_alien(frame_bgr):
    """Transform a frame into an alien with green overlay and antennas."""
    h, w = frame_bgr.shape[:2]
    
    # Extract mask
    mask = extract_mask(frame_bgr)
    
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
    
    # Add antennas
    result = add_antennas(result, mask)
    
    return result


class FrameBuffer:
    """Thread-safe frame buffer that keeps only the latest frame."""
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
    
    def put(self, frame):
        """Store the latest frame (overwrites previous)."""
        with self.lock:
            self.frame = frame
            self.frame_count += 1
    
    def get(self):
        """Get the latest frame."""
        with self.lock:
            return self.frame
    
    def has_frame(self):
        """Check if a frame is available."""
        with self.lock:
            return self.frame is not None


def receive_frames_thread(client_socket, frame_buffer):
    """Thread that continuously receives frames and stores the latest one."""
    print('  Frame receiver started')
    try:
        while True:
            try:
                # Receive frame size
                size_data = client_socket.recv(4)
                if not size_data:
                    print('  Client disconnected (size_data empty)')
                    break
                
                frame_size = int.from_bytes(size_data, 'big')
                
                # Receive frame data
                frame_data = b''
                while len(frame_data) < frame_size:
                    remaining = frame_size - len(frame_data)
                    chunk = client_socket.recv(min(BUFFER_SIZE, remaining))
                    if not chunk:
                        print('  Client disconnected (chunk empty)')
                        return
                    frame_data += chunk
                
                # Decode frame
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame_bgr is not None:
                    frame_buffer.put(frame_bgr)
            
            except Exception as e:
                print(f'  Error receiving frame: {e}')
                break
    
    finally:
        print('  Frame receiver stopped')


def handle_client(client_socket, addr):
    """Handle a connected client - process latest frame each cycle."""
    print(f'Client connected: {addr}')
    
    frame_buffer = FrameBuffer()
    processed_frames = 0
    start_time = time.time()
    
    # Start receiver thread
    receiver_thread = threading.Thread(target=receive_frames_thread, args=(client_socket, frame_buffer))
    receiver_thread.daemon = True
    receiver_thread.start()
    
    print(f'  Waiting for frames...')
    
    try:
        while True:
            # Wait a bit for frames to arrive
            time.sleep(0.01)
            
            # Get the latest frame if available
            frame_bgr = frame_buffer.get()
            
            if frame_bgr is not None:
                try:
                    # Transform into alien (green overlay + antennas)
                    alien_frame = make_alien(frame_bgr)
                    processed_frames += 1
                    
                    # Encode and send back the processed frame
                    _, encoded = cv2.imencode('.jpg', alien_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    encoded_data = encoded.tobytes()
                    
                    # Send frame size + data
                    try:
                        client_socket.send(len(encoded_data).to_bytes(4, 'big'))
                        client_socket.send(encoded_data)
                    except BrokenPipeError:
                        print('  Client disconnected (broken pipe)')
                        break
                    
                    if processed_frames % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = processed_frames / elapsed
                        print(f'Server - Processed {processed_frames} frames | FPS: {fps:.2f}')
                
                except Exception as e:
                    print(f'  Error processing/sending frame: {e}')
                    break
    
    except KeyboardInterrupt:
        print('  Server interrupted')
    except Exception as e:
        print(f'Error with client {addr}: {e}')
    finally:
        print(f'Client disconnected: {addr} (Processed: {processed_frames})')
        client_socket.close()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f'Server listening on {HOST}:{PORT}')
    print('Transforming people into aliens (green overlay + antennas)!')
    
    try:
        while True:
            client_socket, addr = server_socket.accept()
            # Handle each client in a separate thread
            client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
            client_thread.daemon = True
            client_thread.start()
    except KeyboardInterrupt:
        print('\nServer shutting down...')
    finally:
        server_socket.close()


if __name__ == '__main__':
    main()
