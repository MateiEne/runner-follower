import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortFollower:
    """
    Folosește YOLOv8 pentru detecție și DeepSORT pentru tracking cu Re-ID,
    astfel încât să te urmărească doar pe tine, indiferent de aglomerație.
    """
    def __init__(self,
                 model_path='yolo11n.pt',
                 min_bound=0.5,
                 max_bound=0.8,
                 left_bound=0.4,
                 right_bound=0.6):
        # Încarcă YOLOv8
        self.model = YOLO(model_path)
        # Initializează DeepSORT (folosește re-ID model intern)
        self.tracker = DeepSort(max_age=30,
                                nn_budget=70,
                                embedder="mobilenet",  # mobilenet sau tf_efficientnet_lite
                                half=False,
                                nms_max_overlap=1.0)
        # Parametri de bandă
        self.min_bound, self.max_bound = min_bound, max_bound
        self.left_bound, self.right_bound = left_bound, right_bound
        # ID-ul track-ului țintă
        self.target_track_id = None
        print("YOLOv8 + DeepSORT inițializat cu succes.")

    def processImage(self, image_data: bytes) -> str:
        # Decode JPEG în BGR
        img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "None|None"
        vis = img.copy()
        H, W, _ = img.shape

        # 1) rulează detecția YOLOv8
        results = self.model(img, classes=[0], verbose=False)[0]
        bboxes = results.boxes.xyxy.cpu().numpy()      # [[x1,y1,x2,y2], ...]
        confidences = results.boxes.conf.cpu().numpy() # [conf1, conf2, ...]
        # Convertim pentru DeepSORT: list de (tl_x, tl_y, w, h)
        dets = []
        for (x1,y1,x2,y2), conf in zip(bboxes, confidences):
            w, h = x2-x1, y2-y1
            dets.append(([int(x1), int(y1), int(w), int(h)], conf, "person"))

        # 2) actualizează tracker-ul
        tracks = self.tracker.update_tracks(dets, frame=img)

        command = "None|None"
        # 3) ia fiecare track activ
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1,y1,w,h = track.to_ltwh()    # left, top, width, height
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            x2, y2 = x1+w, y1+h

            # Desenează cutia și ID-ul
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 4) la prima detecție salvează-ţi propriul ID
            if self.target_track_id is None:
                self.target_track_id = tid
                print(f"[INFO] Ținta ta are now track ID: {tid}")

            # 5) dacă e track-ul tău, calculează comanda
            if tid == self.target_track_id:
                command = self.check_bounds(vis, W, H, x1, y1, w, h)
                break  # nu ne interesează celelalte track-uri

        # 6) afișăm și trimitem comanda
        cv2.imshow("YOLOv8 + DeepSORT", vis)
        return command

    def check_bounds(self, img, W, H, x, y, w, h) -> str:
        h_cmd = self.check_horizontal(img, W, H, x, w)
        v_cmd = self.check_vertical(img, W, H, y)
        return f"{h_cmd}|{v_cmd}"

    def check_horizontal(self, img, W, H, x, w) -> str:
        lb = int(W * self.left_bound)
        rb = int(W * self.right_bound)
        cv2.line(img, (lb,0), (lb,H), (255,120,0), 2)
        cv2.line(img, (rb,0), (rb,H), (0,120,255), 2)
        desired = (lb+rb)//2
        current = x + w//2
        return f"distance#{desired-current}"

    def check_vertical(self, img, W, H, y) -> str:
        min_y = (H - int(H*self.min_bound))//2
        max_y = (H - int(H*self.max_bound))//2
        cv2.line(img, (0,min_y), (W,min_y), (0,255,255), 2)
        cv2.line(img, (0,max_y), (W,max_y), (0,0,255), 2)
        desired = (min_y+max_y)//2
        return f"distance#{desired-y}"
