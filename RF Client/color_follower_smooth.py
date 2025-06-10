import cv2
import numpy as np
from ultralytics import YOLO

def green_ratio(roi):
    """Returnează procentul de pixeli verzi în ROI."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Interval HSV pentru verde (ajustează după simulator)
    lower = np.array([40, 50, 50])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Adăugăm un epsilon pentru a evita diviziunea la zero
    return mask.sum() / (mask.size + 1e-6)

class ColorFollowerSmooth:
    """
    Follower YOLO + culoare verde cu smoothing prin bounding-box precedent.
    Dacă nu se detectează verde, folosește ultima boxă validă.
    """
    def __init__(self,
                 model_path='yolo11n.pt',
                 min_bound=0.5,
                 max_bound=0.8,
                 left_bound=0.4,
                 right_bound=0.6,
                 green_threshold=0.1):
        self.model = YOLO(model_path)
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.green_threshold = green_threshold
        # Bounding-box precedent cu verde
        self.prev_bbox = None
        print("YOLO + ColorFollowerSmooth inițializat.")

    def processImage(self, image_data: bytes) -> str:
        # Decodare JPEG în BGR
        img = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if img is None:
            return "None|None"
        vis = img.copy()
        H, W, _ = img.shape

        # 1) Detecție YOLO de persoane
        res = self.model(img, classes=[0], verbose=False)[0]
        if not res.boxes:
            # Fără cutii YOLO: dacă avem prev_bbox, continuăm; altfel neutr.
            if self.prev_bbox is None:
                cv2.imshow("Follower", vis)
                return "None|None"
            else:
                best_box = self.prev_bbox
        else:
            # 2) Calcul green_ratio pentru fiecare box
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            best_ratio = 0.0
            best_box = None
            for (x1, y1, x2, y2) in xyxy:
                # Crop ROI valid
                x1_, y1_ = max(x1, 0), max(y1, 0)
                x2_, y2_ = min(x2, W), min(y2, H)
                roi = img[y1_:y2_, x1_:x2_]
                ratio = green_ratio(roi)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_box = (x1_, y1_, x2_, y2_)

            # 3) Smoothing: dacă nu găsim verde, folosim prev_bbox
            if best_box is not None and best_ratio >= self.green_threshold:
                # validăm și actualizăm prev_bbox
                self.prev_bbox = best_box
            else:
                if self.prev_bbox is None:
                    # nici cutie precedentă, nor nici verde
                    cv2.imshow("Follower", vis)
                    return "None|None"
                # folosim ultima cutie precedentă
                best_box = self.prev_bbox

        # 4) Extragem coordonate și desenăm
        x1, y1, x2, y2 = best_box
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis,
                    f"Tracked: {best_box}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        # 5) Calculul comenzii PID (dx|dy)
        command = self.check_bounds(vis, W, H, x1, y1, w, h)
        cv2.imshow("Follower", vis)
        return command

    def check_bounds(self, img, W, H, x, y, w, h) -> str:
        h_cmd = self.check_horizontal(img, W, H, x, w)
        v_cmd = self.check_vertical(img, W, H, y)
        return f"{h_cmd}|{v_cmd}"

    def check_horizontal(self, img, W, H, x, w) -> str:
        lb = int(W * self.left_bound)
        rb = int(W * self.right_bound)
        cv2.line(img, (lb, 0), (lb, H), (255, 120, 0), 2)
        cv2.line(img, (rb, 0), (rb, H), (0, 120, 255), 2)
        desired = (lb + rb) // 2
        current = x + w // 2
        return f"distance#{desired - current}"

    def check_vertical(self, img, W, H, y) -> str:
        min_y = (H - int(H * self.min_bound)) // 2
        max_y = (H - int(H * self.max_bound)) // 2
        cv2.line(img, (0, min_y), (W, min_y), (0, 255, 255), 2)
        cv2.line(img, (0, max_y), (W, max_y), (0, 0, 255), 2)
        desired = (min_y + max_y) // 2
        return f"distance#{desired - y}"
