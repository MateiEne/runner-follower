# venv\Scripts\activate.bat

import socket
import numpy as np
import cv2
import struct
import argparse
# Importăm noua noastră clasă de follower
from follower_ultralytics import BoundedFollowerYoloV8
# from follower_deepsort import DeepSortFollower
from color_follower import ColorFollowerYoloV8
from color_follower_smooth import ColorFollowerSmooth
print("Client Ultralytics (YOLOv8) pornit")

# Parsează argumentele din linia de comandă
parser = argparse.ArgumentParser(description='Follower Simulator Client cu YOLOv8')
parser.add_argument('--min-bound', type=float, default=0.5,
                    help='Limita inferioară ca procent din înălțimea imaginii (0.0 la 1.0)')
parser.add_argument('--max-bound', type=float, default=0.8,
                    help='Limita superioară ca procent din înălțimea imaginii (0.0 la 1.0)')
args = parser.parse_args()

# Creează instanța follower-ului folosind noua clasă și argumentele
print(f"Se utilizează YOLOv8n pentru detecția persoanelor (limite: {args.min_bound}, {args.max_bound})")
# Transmitem limitele direct la inițializare
# follower = DeepSortFollower(min_bound=args.min_bound, max_bound=args.max_bound)
# follower = BoundedFollowerYoloV8(min_bound=args.min_bound, max_bound=args.max_bound)
# follower = ColorFollowerYoloV8(min_bound=args.min_bound, max_bound=args.max_bound)
follower = ColorFollowerSmooth(min_bound=args.min_bound, max_bound=args.max_bound)

# Conectarea la server (codul rămâne identic)
try:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 2737))
    print("Conectat la server")
except socket.error as e:
    print(f"Eroare la conectare: {e}")
    exit()


# Bucla principală (codul rămâne aproape identic)
while True:
    # Verifică dacă s-a apăsat o tastă pentru a ieși
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 este codul ASCII pentru tasta Escape
        break

    # Citește imaginea de la server
    try:
        size_data = client.recv(4)
        if not size_data:
            print("Conexiune închisă de server")
            break

        size = struct.unpack("I", size_data)[0]

        image_data = b""
        while len(image_data) < size:
            packet = client.recv(size - len(image_data))
            if not packet:
                break
            image_data += packet

        if len(image_data) != size:
            print(f"Eroare: se așteptau {size} octeți, dar s-au primit {len(image_data)} octeți")
            continue

        # Obține comanda de la follower
        command = follower.processImage(image_data)
        
        # Trimite comanda către server
        command_bytes = command.encode('utf-8')
        command_size = struct.pack("I", len(command_bytes))
        client.sendall(command_size)
        client.sendall(command_bytes)

    except (ConnectionResetError, BrokenPipeError):
        print("Conexiunea cu serverul a fost pierdută.")
        break
    except Exception as e:
        print(f"A apărut o eroare neașteptată: {e}")
        break

print("Închidere conexiune")
client.close()
cv2.destroyAllWindows()