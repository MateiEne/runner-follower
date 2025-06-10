import cv2
import numpy as np
# Importăm clasa YOLO din biblioteca ultralytics
from ultralytics import YOLO

class BoundedFollowerYoloV8():
    """
    Un follower care detectează o persoană folosind YOLOv8,
    desenează un chenar și controale vizuale, și generează comenzi.
    """
    def __init__(self,
                 model_path='yolo11n.pt',  # Modelul nano, echivalentul modern al 'tiny'
                 min_bound=0.5,
                 max_bound=0.8,
                 left_bound=0.4,
                 right_bound=0.6):
        """
        Inițializează detectorul YOLOv11.
        Biblioteca ultralytics va descărca automat 'yolo11n.pt' la prima rulare.
        """
        # 1. Încărcarea modelului - o singură linie de cod!
        self.model = YOLO(model_path)
        
        # Salvează limitele pentru control
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.left_bound = left_bound
        self.right_bound = right_bound
        
        print("Modelul YOLOv8 a fost încărcat cu succes.")

    def processImage(self, image_data):
        """
        Procesează imaginea, detectează persoane și returnează o comandă.
        """
        # Decodează datele imaginii primite de la server
        image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return "None|None" # Returnează o comandă neutră în caz de eroare

        # Creează o copie a imaginii pentru a desena pe ea
        result_image = image.copy()
        height, width, _ = result_image.shape

        # 2. Rularea detecției - o singură linie de cod!
        # 'classes=[0]' -> îi spunem să caute DOAR persoane (clasa 0 în COCO)
        # 'verbose=False' -> nu afișează informații de debug în consolă
        results = self.model(image, classes=[0], verbose=False)

        command = "None|None"

        # 3. Procesarea rezultatelor - mult mai simplu!
        # Rezultatele sunt deja filtrate și sortate
        if results and len(results[0].boxes) > 0:
            # Luăm prima persoană detectată (cea cu cea mai mare încredere)
            box = results[0].boxes[0]
            
            # Extragem coordonatele (x1, y1, x2, y2)
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            w, h = x2 - x1, y2 - y1
            
            # Extragem încrederea
            confidence = box.conf[0]
            
            # --- Vizualizare (Desenare pe imagine) ---
            # Desenează chenarul în jurul persoanei
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adaugă eticheta cu informații
            label = f"Person: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Generează comanda bazată pe poziția persoanei
            command = self.check_bounds(result_image, width, height, x1, y1, w, h)
        
        # Afișează imaginea rezultată
        cv2.imshow("YOLOv8 Follower", result_image)
        
        return command

    def check_bounds(self, result_image, width, height, x, y, w, h):
        """
        Combină comenzile orizontale și verticale.
        """
        horizontal_command = self.check_horizontal_bounds(result_image, width, height, x, w)
        vertical_command = self.check_vertical_bounds(result_image, width, height, y)
        return f"{horizontal_command}|{vertical_command}"

    def check_horizontal_bounds(self, result_image, width, height, x, w):
        """
        Calculează comanda de mișcare orizontală ('stânga'/'dreapta').
        (Acest cod este identic cu cel original, deoarece logica de control nu s-a schimbat)
        """
        left_bound_x = int(width * self.left_bound)
        right_bound_x = int(width * self.right_bound)
        
        cv2.line(result_image, (left_bound_x, 0), (left_bound_x, height), (255, 120, 0), 2)
        cv2.line(result_image, (right_bound_x, 0), (right_bound_x, height), (0, 120, 255), 2)
 
        desired_position = int((left_bound_x + right_bound_x) / 2)
        current_position = x + int(w / 2)
        
        return f"distance#{desired_position - current_position}"

    def check_vertical_bounds(self, result_image, width, height, y):
        """
        Calculează comanda de mișcare verticală ('înainte'/'înapoi').
        (Acest cod este identic cu cel original)
        """
        min_rect_height = int(height * self.min_bound)
        min_rect_y = int((height - min_rect_height) / 2)
        cv2.line(result_image, (0, min_rect_y), (width, min_rect_y), (0, 255, 255), 2)
        
        max_rect_height = int(height * self.max_bound)
        max_rect_y = int((height - max_rect_height) / 2)
        cv2.line(result_image, (0, max_rect_y), (width, max_rect_y), (0, 0, 255), 2)

        desired_position = int((min_rect_y + max_rect_y) / 2)
        current_position = y
        
        return f"distance#{desired_position - current_position}"