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


# follower = DeepSortFollower(min_bound=args.min_bound, max_bound=args.max_bound)
# follower = BoundedFollowerYoloV8(min_bound=args.min_bound, max_bound=args.max_bound)
# follower = ColorFollowerYoloV8(min_bound=args.min_bound, max_bound=args.max_bound)
follower = ColorFollowerSmooth()

try:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 2737))
    print("Conectat la server")
except socket.error as e:
    print(f"Eroare la conectare: {e}")
    exit()


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 este codul ASCII pentru tasta Escape
        break

    # Citeste imaginea de la server
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

        # Obtine comanda de la follower
        command = follower.processImage(image_data)
        
        # Trimite comanda catre server
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