import socket
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time

# Configuration
HOST = '127.0.0.1'
PORT = 5000
BUFFER_SIZE = 65536

# Model configuration
IMG_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Server device: {device}')


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


# Load model
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

# Load background
print('Loading background...')
background = cv2.imread('background.png')
if background is None:
    print('Warning: background.png not found')
else:
    print('Background loaded!')


def segment_person(frame_bgr, bg):
    """Segment a person from a frame and return the processed frame with background."""
    h, w = frame_bgr.shape[:2]
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.tensor(img_normalized.transpose(2, 0, 1), 
                              dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()
    
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
    
    # Apply mask to frame with background
    if bg is not None:
        bg_resized = cv2.resize(bg, (w, h))
    else:
        bg_resized = np.full_like(frame_bgr, 255)
    
    # Apply mask: keep original where mask is white, use background elsewhere
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch > 127, frame_bgr, bg_resized)
    
    return result.astype(np.uint8)


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
                    # Process the latest frame with background
                    processed_frame = segment_person(frame_bgr, background)
                    processed_frames += 1
                    
                    # Encode and send back the processed frame
                    _, encoded = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
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
    print('Always processing the latest frame received')
    
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
