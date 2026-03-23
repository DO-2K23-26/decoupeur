import socket
import cv2
import numpy as np
import time
import threading

# Configuration
SERVER_HOST = '162.38.112.84'
#SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000
BUFFER_SIZE = 65536
TARGET_FPS = 60

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.cap = None
        self.received_frame = None
        self.lock = threading.Lock()
    
    def connect(self):
        """Connect to the server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f'Connected to server at {self.host}:{self.port}')
        self.running = True
    
    def send_frames(self):
        """Send frames to the server."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_number = 0
        start_time = time.time()
        frame_time = 1.0 / TARGET_FPS  # Time per frame for 60 FPS
        
        print(f'Sending frames at {TARGET_FPS} FPS...')
        
        try:
            while self.running:
                loop_start = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print('Failed to capture frame')
                    break
                
                # Flip horizontal to correct mirror effect
                frame = cv2.flip(frame, 1)
                
                try:
                    # Encode frame
                    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    encoded_data = encoded.tobytes()
                    
                    # Send frame size + data
                    self.socket.send(len(encoded_data).to_bytes(4, 'big'))
                    self.socket.send(encoded_data)
                    
                    frame_number += 1
                    if frame_number % 60 == 0:
                        elapsed = time.time() - start_time
                        actual_fps = frame_number / elapsed
                        print(f'Client - Sent {frame_number} frames | Target FPS: {TARGET_FPS} | Actual FPS: {actual_fps:.2f}')
                    
                    # Maintain 60 FPS
                    loop_time = time.time() - loop_start
                    sleep_time = frame_time - loop_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                except Exception as e:
                    print(f'Error sending frame: {e}')
                    break
        
        finally:
            if self.cap:
                self.cap.release()
            print('Frame sender stopped')
    
    def receive_frames(self):
        """Receive processed frames from the server."""
        frame_number = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Receive frame size
                size_data = self.socket.recv(4)
                if not size_data:
                    break
                
                frame_size = int.from_bytes(size_data, 'big')
                
                # Receive frame data
                frame_data = b''
                while len(frame_data) < frame_size:
                    chunk = self.socket.recv(min(BUFFER_SIZE, frame_size - len(frame_data)))
                    if not chunk:
                        break
                    frame_data += chunk
                
                if len(frame_data) < frame_size:
                    break
                
                # Decode frame
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with self.lock:
                        self.received_frame = frame
                    
                    frame_number += 1
                    if frame_number % 60 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_number / elapsed
                        print(f'Display - Received {frame_number} frames | FPS: {fps:.2f}')
        
        except Exception as e:
            print(f'Error receiving frames: {e}')
        finally:
            print('Frame receiver stopped')
    
    def display_frames(self):
        """Display received processed frames."""
        window_name = 'Segmented Feed'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        fullscreen = False
        screen_width, screen_height = 1920, 1080
        
        print('Display window started')
        print('  Q to quit')
        print('  F to toggle fullscreen')
        
        try:
            while self.running:
                with self.lock:
                    if self.received_frame is not None:
                        frame_to_display = self.received_frame
                        
                        if fullscreen:
                            frame_to_display = cv2.resize(self.received_frame, (screen_width, screen_height))
                        
                        cv2.imshow(window_name, frame_to_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print('Fullscreen ON')
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print('Fullscreen OFF')
                
                time.sleep(0.001)
        
        finally:
            cv2.destroyAllWindows()
            print('Display stopped')
    
    def close(self):
        """Close the connection."""
        self.running = False
        if self.socket:
            self.socket.close()
        print('Connection closed')


def main():
    client = VideoClient(SERVER_HOST, SERVER_PORT)
    
    try:
        client.connect()
        
        # Start sender thread
        sender_thread = threading.Thread(target=client.send_frames, daemon=True)
        sender_thread.start()
        
        # Start receiver thread
        receiver_thread = threading.Thread(target=client.receive_frames, daemon=True)
        receiver_thread.start()
        
        # Display frames in main thread
        client.display_frames()
    
    except Exception as e:
        print(f'Client error: {e}')
    finally:
        client.close()


if __name__ == '__main__':
    main()
