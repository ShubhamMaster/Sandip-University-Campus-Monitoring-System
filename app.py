"""
People Counting Web Application
Flask + YOLOv8 + CentroidTracker
"""

import json
import csv
import os
import time
import datetime
import threading
import logging
from itertools import zip_longest

import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from utils.mailer import Mailer
from utils.thread import ThreadingClass

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "people-counter-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Load config
with open("utils/config.json", "r") as f:
    config = json.load(f)

# Ensure log directory exists
os.makedirs("utils/data/logs", exist_ok=True)

# ---------------------------------------------------------------------------
# Global state shared across streams
# ---------------------------------------------------------------------------
class CounterState:
    """Holds per-camera counting state."""
    def __init__(self):
        self.total_up = 0        # exited
        self.total_down = 0      # entered
        self.total_inside = 0
        self.status = "Idle"
        self.threshold = config.get("Threshold", 10)
        self.threshold_exceeded = False
        # logging lists
        self.move_in = []
        self.move_out = []
        self.in_time = []
        self.out_time = []
        # occupancy history for chart
        self.occupancy_history = []   # list of {"time": str, "count": int}
        # Crowd-rush detection: track entry timestamps in a sliding window
        self.rush_limit = config.get("RushLimit", 10)
        self.rush_window = config.get("RushWindow", 10)  # seconds
        self.entry_timestamps = []  # list of datetime objects
        self.rush_detected = False
        # Detection data for web-based canvas overlay
        self.current_detections = []   # [{"id": int, "cx": int, "cy": int, "cls": str}]
        self.current_rects = []        # [[x1,y1,x2,y2], ...]
        self.current_rect_labels = []  # ["Person", "Car", ...]
        self.frame_size = [0, 0]       # [W, H]
        self.line_y = 0
        self.gate_lines = []           # [{"label": "OUT1"|"GATE"|"IN1", "y": int, "lvl": int}, ...]
        self.frame_seq = -1
        self.type_counts_in = {}       # {"Person": 3, "Car": 1, ...}
        self.type_counts_out = {}      # {"Person": 2, ...}
        self.lock = threading.Lock()

# active camera streams: camera_index -> dict with cap, state, running flag
active_streams: dict[int, dict] = {}
streams_lock = threading.Lock()

# Cached camera list (avoid slow re-probing)
_camera_cache: list[dict] = []
_camera_cache_time: float = 0

# Per-camera video transforms (rotation, flip)
# camera_index -> {"rotation": 0|90|180|270, "flip_h": bool, "flip_v": bool}
_video_transforms: dict[int, dict] = {}
_transforms_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helper – discover cameras
# ---------------------------------------------------------------------------
def _try_open_camera(idx: int) -> cv2.VideoCapture | None:
    """Try multiple backends to open a camera index."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(idx, backend)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return None

def _get_camera_device_names() -> list[str]:
    """Try to get friendly camera device names from Windows WMI."""
    try:
        import subprocess
        result = subprocess.run(
            'wmic path Win32_PnPEntity where "PNPClass=\'Camera\' or PNPClass=\'Image\'" get Name /format:list',
            capture_output=True, text=True, timeout=5, shell=True,
        )
        names = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('Name='):
                names.append(line[5:].strip())
        return names
    except Exception:
        return []

def discover_cameras(max_index: int = 5, force: bool = False) -> list[dict]:
    """Probe camera indices and return available ones. Results cached for 60s."""
    global _camera_cache, _camera_cache_time
    if not force and _camera_cache and (time.time() - _camera_cache_time) < 60:
        return _camera_cache
    device_names = _get_camera_device_names()
    available = []
    for idx in range(max_index):
        cap = _try_open_camera(idx)
        if cap is not None:
            if idx < len(device_names) and device_names[idx]:
                friendly = device_names[idx]
            else:
                friendly = "Built-in Webcam" if idx == 0 else f"External Camera {idx}"
            available.append({
                "index": idx,
                "name": friendly,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            })
            cap.release()
    _camera_cache = available
    _camera_cache_time = time.time()
    return available

# ---------------------------------------------------------------------------
# YOLOv8 model (lazy loaded)
# ---------------------------------------------------------------------------
_yolo_model = None
_yolo_lock = threading.Lock()

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        with _yolo_lock:
            if _yolo_model is None:
                model_name = config.get("yolo_model", "yolov8n.pt")
                logger.info(f"Loading YOLOv8 model: {model_name}")
                _yolo_model = YOLO(model_name)
                logger.info("YOLOv8 model loaded.")
    return _yolo_model

# ---------------------------------------------------------------------------
# Email helper
# ---------------------------------------------------------------------------
def send_mail():
    try:
        Mailer().send(config["Email_Receive"])
    except Exception as e:
        logger.error(f"Email send failed: {e}")

# ---------------------------------------------------------------------------
# Data logging
# ---------------------------------------------------------------------------
def log_data(move_in, in_time, move_out, out_time):
    data = [move_in, in_time, move_out, out_time]
    export_data = zip_longest(*data, fillvalue="")
    with open("utils/data/logs/counting_data.csv", "w", newline="") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
        wr.writerows(export_data)

# ---------------------------------------------------------------------------
# Frame generator – processes video with YOLOv8 + CentroidTracker
# ---------------------------------------------------------------------------
# COCO class IDs we want to detect (person + vehicles)
DETECT_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Target FPS for streaming
TARGET_FPS = 30
FRAME_W = 1280  # 720p width (16:9)
FRAME_H = 720   # 720p height (16:9)

def _emit_detections(camera_index: int, state):
    """Push detection overlay data via Socket.IO."""
    with state.lock:
        socketio.emit('detections', {
            'd': state.current_detections,
            'r': state.current_rects,
            's': state.frame_size,
            'ly': state.line_y,
            'lys': state.gate_lines,
            'seq': state.frame_seq,
        }, namespace='/')

def _emit_stats(camera_index: int, state):
    """Push stats via Socket.IO."""
    with state.lock:
        now_dt = datetime.datetime.now()
        cutoff = now_dt - datetime.timedelta(seconds=state.rush_window)
        state.entry_timestamps = [t for t in state.entry_timestamps if t > cutoff]
        state.rush_detected = len(state.entry_timestamps) >= state.rush_limit
        socketio.emit('stats', {
            'entered': state.total_down,
            'exited': state.total_up,
            'inside': state.total_inside,
            'status': state.status,
            'threshold_exceeded': state.threshold_exceeded,
            'threshold': state.threshold,
            'rush_detected': state.rush_detected,
            'rush_limit': state.rush_limit,
            'rush_window': state.rush_window,
            'recent_entries': len(state.entry_timestamps),
            'type_counts_in': state.type_counts_in.copy(),
            'type_counts_out': state.type_counts_out.copy(),
            'type_inside': {
                cls: max(0, state.type_counts_in.get(cls, 0) - state.type_counts_out.get(cls, 0))
                for cls in set(list(state.type_counts_in.keys()) + list(state.type_counts_out.keys()))
                if state.type_counts_in.get(cls, 0) - state.type_counts_out.get(cls, 0) > 0
            },
        }, namespace='/')

def _emit_chart(camera_index: int, state):
    """Push occupancy chart data via Socket.IO."""
    with state.lock:
        socketio.emit('chart', state.occupancy_history.copy(), namespace='/')

def generate_frames(camera_index: int):
    """Generator that yields MJPEG frames for a given camera."""

    model = get_yolo_model()
    confidence = config.get("confidence", 0.4)
    line_pos_ratio = float(config.get("LinePositionRatio", 0.5))
    line_pos_ratio = max(0.15, min(0.85, line_pos_ratio))
    gate_lines_per_side = int(config.get("GateLinesPerSide", 4))
    gate_lines_per_side = max(1, min(6, gate_lines_per_side))
    # Process detection on a lower internal resolution to reduce CPU/GPU load.
    process_h = int(config.get("ProcessHeight", 480))
    process_h = max(240, min(720, process_h))
    preview_max_h = int(config.get("PreviewMaxHeight", 1080))
    preview_max_h = max(480, min(2160, preview_max_h))
    detect_imgsz = int(config.get("DetectImgSize", 480))
    detect_imgsz = max(320, min(960, detect_imgsz))
    detection_max_fps = float(config.get("DetectionMaxFPS", 30))
    detection_max_fps = max(3.0, min(30.0, detection_max_fps))
    detection_stride = int(config.get("DetectionStride", 1))
    detection_stride = max(1, min(10, detection_stride))
    detection_interval = 1.0 / detection_max_fps

    # Open the camera via threading class for reduced lag
    if config.get("Thread", False):
        vs = ThreadingClass(camera_index)
    else:
        vs = _try_open_camera(camera_index)
        if vs is None:
            logger.error(f"Cannot open camera {camera_index}")
            return

    ct = CentroidTracker(maxDisappeared=15, maxDistance=70)
    trackableObjects = {}
    state = CounterState()

    # Stop ALL other streams so only the selected camera runs (reduces lag)
    with streams_lock:
        for idx in list(active_streams.keys()):
            active_streams[idx]["running"] = False
        time.sleep(0.2)
        active_streams.clear()
        active_streams[camera_index] = {
            "state": state,
            "running": True,
        }

    W, H = None, None
    object_classes = {}
    object_class_votes = {}
    object_rects = {}  # objectID -> (x1, y1, x2, y2) last matched bbox
    object_matched = {}  # objectID -> bool (matched to a rect this frame)
    object_events = {}  # objectID -> None | "IN" | "OUT"
    object_gate_state = {}  # objectID -> {"level": int, "count_side": int, "last_cy": int, "last_event_ts": float}
    frame_interval = 1.0 / TARGET_FPS

    def _build_gate_lines(frame_h: int):
        gate_y = int(frame_h * line_pos_ratio)
        gate_y = int(max(0, min(frame_h - 1, gate_y)))
        out_span = max(1.0, float(gate_y - 0))
        in_span = max(1.0, float((frame_h - 1) - gate_y))
        out_step = max(1.0, out_span / float(gate_lines_per_side))
        in_step = max(1.0, in_span / float(gate_lines_per_side))
        lines = []
        for lvl in range(-gate_lines_per_side, gate_lines_per_side + 1):
            if lvl < 0:
                y = int(round(gate_y - (abs(lvl) * out_step)))
            elif lvl > 0:
                y = int(round(gate_y + (lvl * in_step)))
            else:
                y = gate_y
            y = int(max(0, min(frame_h - 1, y)))
            if lvl < 0:
                label = f"OUT{abs(lvl)}"
            elif lvl > 0:
                label = f"IN{lvl}"
            else:
                label = "GATE"
            lines.append({"label": label, "y": y, "lvl": lvl})
        return lines, gate_y, out_step, in_step

    def _level_from_y(cy: int, gate_y: int, out_step: float, in_step: float):
        gate_band = max(3, int(round(min(out_step, in_step) * 0.30)))
        if abs(cy - gate_y) <= gate_band:
            return 0
        if cy < gate_y:
            dist = gate_y - cy
            lvl = int(round(dist / max(1.0, out_step)))
            lvl = max(1, min(gate_lines_per_side, lvl))
            return -lvl
        dist = cy - gate_y
        lvl = int(round(dist / max(1.0, in_step)))
        lvl = max(1, min(gate_lines_per_side, lvl))
        return lvl

    def _apply_count_event(state_obj: CounterState, obj_id: int, event: str, obj_cls: str):
        now_dt = datetime.datetime.now()
        dt = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        if event == "OUT":
            state_obj.total_up += 1
            state_obj.move_out.append(state_obj.total_up)
            state_obj.out_time.append(dt)
            state_obj.type_counts_out[obj_cls] = state_obj.type_counts_out.get(obj_cls, 0) + 1
        elif event == "IN":
            state_obj.total_down += 1
            state_obj.move_in.append(state_obj.total_down)
            state_obj.in_time.append(dt)
            state_obj.type_counts_in[obj_cls] = state_obj.type_counts_in.get(obj_cls, 0) + 1
            state_obj.entry_timestamps.append(now_dt)
            cutoff = now_dt - datetime.timedelta(seconds=state_obj.rush_window)
            state_obj.entry_timestamps = [t for t in state_obj.entry_timestamps if t > cutoff]
            state_obj.rush_detected = len(state_obj.entry_timestamps) >= state_obj.rush_limit

    def _sign(v: int):
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    # --- Shared frame buffer for background detection ---
    _shared_frame = [None]
    _shared_seq = [-1]
    _shared_lock = threading.Lock()
    _worker_alive = [True]

    def _detection_worker():
        """Background thread: runs YOLO + tracker continuously, emits Socket.IO."""
        nonlocal W, H
        det_count = 0
        _last_occ = time.time()
        _last_seq = -1
        _last_det_ts = 0.0
        while _worker_alive[0] and active_streams.get(camera_index, {}).get("running", False):
            with _shared_lock:
                frame = _shared_frame[0]
                frame_seq = _shared_seq[0]
            if frame is None:
                time.sleep(0.02)
                continue
            if frame_seq == _last_seq:
                time.sleep(0.005)
                continue
            _last_seq = frame_seq

            if detection_stride > 1 and (frame_seq % detection_stride) != 0:
                continue

            now_ts = time.time()
            if (now_ts - _last_det_ts) < detection_interval:
                time.sleep(0.001)
                continue
            _last_det_ts = now_ts

            (H, W) = frame.shape[:2]
            state.status = "Detecting"

            # Downscale only for detection; preview keeps source quality.
            proc_h_cur = min(process_h, H)
            proc_w_cur = max(2, int(round(proc_h_cur * (W / float(H)))))
            proc_frame = cv2.resize(frame, (proc_w_cur, proc_h_cur), interpolation=cv2.INTER_LINEAR)
            sx = W / float(proc_w_cur)
            sy = H / float(proc_h_cur)

            results = model(proc_frame, imgsz=detect_imgsz, classes=list(DETECT_CLASSES.keys()), verbose=False)
            rects = []
            rect_labels = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in DETECT_CLASSES and conf >= confidence:
                        x1p, y1p, x2p, y2p = box.xyxy[0].cpu().numpy()
                        x1 = int(max(0, min(W - 1, x1p * sx)))
                        y1 = int(max(0, min(H - 1, y1p * sy)))
                        x2 = int(max(0, min(W - 1, x2p * sx)))
                        y2 = int(max(0, min(H - 1, y2p * sy)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        rects.append((x1, y1, x2, y2))
                        rect_labels.append(DETECT_CLASSES[cls_id])

            state.status = "Tracking"
            objects = ct.update(rects)

            # Build exact centroid→rect-index map (tracker sets object centroid
            # = input centroid for matched objects, so exact match = matched)
            _input_cents = {}
            for _ri, (_rx1, _ry1, _rx2, _ry2) in enumerate(rects):
                _ck = (int((_rx1+_rx2)/2), int((_ry1+_ry2)/2))
                _input_cents[_ck] = _ri

            with state.lock:
                gate_lines, gate_y, out_step, in_step = _build_gate_lines(H)
                for (objectID, centroid) in objects.items():
                    # Find which rect this object was matched to by the tracker
                    _ckey = (int(centroid[0]), int(centroid[1]))
                    matched_ri = _input_cents.get(_ckey)

                    to = trackableObjects.get(objectID, None)
                    if to is None:
                        to = TrackableObject(objectID, centroid)
                        if matched_ri is not None and matched_ri < len(rect_labels):
                            label = rect_labels[matched_ri]
                        else:
                            label = "Unidentified"
                        # Start as Unidentified until we gather enough class votes.
                        object_classes[objectID] = "Unidentified"
                        object_class_votes[objectID] = {label: 1}
                        object_rects[objectID] = rects[matched_ri] if matched_ri is not None else None
                        object_matched[objectID] = matched_ri is not None
                        object_events[objectID] = None
                    else:
                        # Only update class vote when tracker actually matched to a rect
                        if matched_ri is not None and matched_ri < len(rect_labels):
                            det_label = rect_labels[matched_ri]
                            votes = object_class_votes.setdefault(objectID, {})
                            votes[det_label] = votes.get(det_label, 0) + 1
                            total_votes = sum(votes.values())
                            top_label = max(votes, key=votes.get)
                            top_votes = votes[top_label]
                            # Label only when vote support is strong enough.
                            if total_votes >= 3 and (top_votes / total_votes) >= 0.60:
                                object_classes[objectID] = top_label
                            else:
                                object_classes[objectID] = "Unidentified"
                            object_rects[objectID] = rects[matched_ri]
                            object_matched[objectID] = True
                        else:
                            # Not matched this frame — mark as unmatched
                            object_matched[objectID] = False

                        to.centroids.append(centroid)
                        cur_level = _level_from_y(int(centroid[1]), gate_y, out_step, in_step)

                        gs = object_gate_state.get(objectID)
                        if gs is None:
                            object_gate_state[objectID] = {
                                "level": cur_level,
                                "count_side": 0,
                                "last_cy": int(centroid[1]),
                                "last_event_ts": 0.0,
                            }
                        else:
                            prev_level = int(gs.get("level", cur_level))
                            prev_side = _sign(prev_level)
                            cur_side = _sign(cur_level)
                            moved_away = abs(cur_level) > abs(prev_level)
                            now_ts = time.time()
                            cooldown_ok = (now_ts - float(gs.get("last_event_ts", 0.0))) > 0.9
                            event = None

                            # Rule 1: crossing gate to first line on either side counts immediately.
                            if cooldown_ok and cur_side != 0 and cur_side != prev_side:
                                event = "IN" if cur_side > 0 else "OUT"

                            # Rule 2: if gate crossing was missed, crossing deeper side lines also counts.
                            elif (
                                cooldown_ok
                                and cur_side != 0
                                and moved_away
                                and abs(cur_level) >= 2
                                and int(gs.get("count_side", 0)) != cur_side
                            ):
                                event = "IN" if cur_side > 0 else "OUT"

                            if event is not None:
                                obj_cls = object_classes.get(objectID, "Person")
                                _apply_count_event(state, objectID, event, obj_cls)
                                object_events[objectID] = event
                                gs["count_side"] = cur_side
                                gs["last_event_ts"] = now_ts

                            # Reset side lock when object comes back to gate zone.
                            if cur_side == 0:
                                gs["count_side"] = 0

                            gs["level"] = cur_level
                            gs["last_cy"] = int(centroid[1])

                        state.total_inside = max(0, len(state.move_in) - len(state.move_out))

                        if state.total_inside >= state.threshold:
                            if not state.threshold_exceeded and config.get("ALERT"):
                                threading.Thread(target=send_mail, daemon=True).start()
                            state.threshold_exceeded = True
                        else:
                            state.threshold_exceeded = False

                    trackableObjects[objectID] = to

                for oid in list(object_classes.keys()):
                    if oid not in objects:
                        del object_classes[oid]
                        object_class_votes.pop(oid, None)
                        object_rects.pop(oid, None)
                        object_matched.pop(oid, None)
                        object_events.pop(oid, None)
                        object_gate_state.pop(oid, None)

                # Only emit actively-matched objects (those with a real bbox)
                det_list = []
                det_rects = []
                for (oid, ctr) in objects.items():
                    if not object_matched.get(oid, False):
                        continue  # skip disappeared/unmatched — no visual
                    cls = object_classes.get(oid, "Unidentified")
                    r = object_rects.get(oid)
                    if r:
                        det_list.append({
                            "id": int(oid),
                            "cx": int(ctr[0]),
                            "cy": int(ctr[1]),
                            "cls": cls,
                            "ev": object_events.get(oid),
                        })
                        det_rects.append([int(r[0]), int(r[1]), int(r[2]), int(r[3])])
                state.current_detections = det_list
                state.current_rects = det_rects
                state.current_rect_labels = [d["cls"] for d in det_list]
                state.frame_size = [W, H]
                state.line_y = gate_y
                state.gate_lines = gate_lines
                state.frame_seq = int(frame_seq)

            det_count += 1

            # Occupancy history every ~2s
            _now = time.time()
            if _now - _last_occ >= 2.0:
                _last_occ = _now
                with state.lock:
                    state.occupancy_history.append({
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "count": state.total_inside,
                    })
                    if len(state.occupancy_history) > 300:
                        state.occupancy_history = state.occupancy_history[-300:]

            if config.get("Log") and det_count % 30 == 0:
                log_data(state.move_in, state.in_time, state.move_out, state.out_time)

            # Socket.IO emit
            _emit_detections(camera_index, state)
            if det_count % 4 == 0:
                _emit_stats(camera_index, state)
            if det_count % 15 == 0:
                _emit_chart(camera_index, state)

    # Spawn background detection worker
    _det_thread = threading.Thread(target=_detection_worker, daemon=True)
    _det_thread.start()

    totalFrames = 0
    try:
        while active_streams.get(camera_index, {}).get("running", False):
            t_start = time.time()

            # Read frame
            if config.get("Thread", False):
                frame = vs.read()
            else:
                ret, frame = vs.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            if frame is None:
                time.sleep(0.05)
                continue

            # Apply video transforms (rotation/flip) BEFORE resize & detection
            with _transforms_lock:
                tfm = _video_transforms.get(camera_index, {})
            rot = tfm.get("rotation", 0)
            if rot == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if tfm.get("flip_h", False):
                frame = cv2.flip(frame, 1)
            if tfm.get("flip_v", False):
                frame = cv2.flip(frame, 0)

            # Keep preview close to source quality (e.g., 720p/1080p), cap very high feeds.
            h0, w0 = frame.shape[:2]
            if h0 > preview_max_h:
                scale = preview_max_h / float(h0)
                new_w = max(2, int(round(w0 * scale)))
                frame = cv2.resize(frame, (new_w, preview_max_h), interpolation=cv2.INTER_LINEAR)

            # Enforce 16:9 output ratio for preview and overlay alignment.
            h1, w1 = frame.shape[:2]
            target_ratio = 16.0 / 9.0
            current_ratio = w1 / float(h1)
            if current_ratio > target_ratio:
                # Too wide: crop width from center.
                new_w = int(round(h1 * target_ratio))
                x0 = max(0, (w1 - new_w) // 2)
                frame = frame[:, x0:x0 + new_w]
            elif current_ratio < target_ratio:
                # Too tall: crop height from center.
                new_h = int(round(w1 / target_ratio))
                y0 = max(0, (h1 - new_h) // 2)
                frame = frame[y0:y0 + new_h, :]

            # Feed to detection worker (non-blocking)
            with _shared_lock:
                _shared_frame[0] = frame
                _shared_seq[0] = totalFrames

            totalFrames += 1

            # Encode & yield MJPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            # FPS limiter
            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    finally:
        _worker_alive[0] = False
        _det_thread.join(timeout=2)
        if config.get("Thread", False):
            vs.release()
        else:
            vs.release()
        with streams_lock:
            active_streams.pop(camera_index, None)
        logger.info(f"Camera {camera_index} stream stopped.")

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        threshold=config.get("Threshold", 10),
        rush_limit=config.get("RushLimit", 10),
        rush_window=config.get("RushWindow", 10),
    )

@app.route("/api/cameras")
def api_cameras():
    """Return list of available cameras."""
    cameras = discover_cameras(max_index=5)
    return jsonify(cameras)

@app.route("/video_feed/<int:camera_index>")
def video_feed(camera_index):
    """MJPEG streaming route."""
    return Response(
        generate_frames(camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/api/stats/<int:camera_index>")
def api_stats(camera_index):
    """Return real-time counter stats for a camera."""
    with streams_lock:
        stream = active_streams.get(camera_index)
    if stream is None:
        return jsonify({
            "entered": 0, "exited": 0, "inside": 0,
            "status": "Idle", "threshold_exceeded": False,
            "threshold": config.get("Threshold", 10),
        })
    state = stream["state"]
    with state.lock:
        # Refresh rush detection even when no new entry
        now_dt = datetime.datetime.now()
        cutoff = now_dt - datetime.timedelta(seconds=state.rush_window)
        state.entry_timestamps = [t for t in state.entry_timestamps if t > cutoff]
        state.rush_detected = len(state.entry_timestamps) >= state.rush_limit
        return jsonify({
            "entered": state.total_down,
            "exited": state.total_up,
            "inside": state.total_inside,
            "status": state.status,
            "threshold_exceeded": state.threshold_exceeded,
            "threshold": state.threshold,
            "rush_detected": state.rush_detected,
            "rush_limit": state.rush_limit,
            "rush_window": state.rush_window,
            "recent_entries": len(state.entry_timestamps),
            "type_counts_in": state.type_counts_in.copy(),
            "type_counts_out": state.type_counts_out.copy(),
            "type_inside": {
                cls: max(0, state.type_counts_in.get(cls, 0) - state.type_counts_out.get(cls, 0))
                for cls in set(list(state.type_counts_in.keys()) + list(state.type_counts_out.keys()))
                if state.type_counts_in.get(cls, 0) - state.type_counts_out.get(cls, 0) > 0
            },
            # Detection overlay data for web canvas
            "detections": state.current_detections,
            "rects": state.current_rects,
            "frame_size": state.frame_size,
        })

@app.route("/api/occupancy/<int:camera_index>")
def api_occupancy(camera_index):
    """Return occupancy history for chart."""
    with streams_lock:
        stream = active_streams.get(camera_index)
    if stream is None:
        return jsonify([])
    state = stream["state"]
    with state.lock:
        return jsonify(state.occupancy_history.copy())

@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    """Dynamically update threshold."""
    data = request.get_json()
    new_threshold = data.get("threshold", 10)
    config["Threshold"] = int(new_threshold)
    # Update all active streams
    with streams_lock:
        for cam_data in active_streams.values():
            cam_data["state"].threshold = int(new_threshold)
    # Persist to config
    with open("utils/config.json", "w") as f:
        json.dump(config, f, indent=4)
    return jsonify({"success": True, "threshold": int(new_threshold)})

@app.route("/api/rush_limit", methods=["POST"])
def api_rush_limit():
    """Dynamically update crowd-rush limit and window."""
    data = request.get_json()
    new_limit = int(data.get("rush_limit", 10))
    new_window = int(data.get("rush_window", 10))
    config["RushLimit"] = new_limit
    config["RushWindow"] = new_window
    with streams_lock:
        for cam_data in active_streams.values():
            cam_data["state"].rush_limit = new_limit
            cam_data["state"].rush_window = new_window
    with open("utils/config.json", "w") as f:
        json.dump(config, f, indent=4)
    return jsonify({"success": True, "rush_limit": new_limit, "rush_window": new_window})

@app.route("/api/stop/<int:camera_index>", methods=["POST"])
def api_stop(camera_index):
    """Stop a camera stream."""
    with streams_lock:
        if camera_index in active_streams:
            active_streams[camera_index]["running"] = False
    return jsonify({"success": True})

@app.route("/api/detections/<int:camera_index>")
def api_detections(camera_index):
    """Fast lightweight endpoint — returns only bounding boxes + centroids
    for the canvas overlay.  Polled at ~100-150 ms from the browser."""
    with streams_lock:
        stream = active_streams.get(camera_index)
    if stream is None:
        return jsonify({"d": [], "r": [], "s": [0, 0]})
    st = stream["state"]
    with st.lock:
        return jsonify({
            "d": st.current_detections,    # [{id, cx, cy}]
            "r": st.current_rects,          # [[x1,y1,x2,y2]]
            "s": st.frame_size,             # [W, H]
        })

@app.route("/api/video_transform/<int:camera_index>")
def api_get_transform(camera_index):
    """Return current video transform for a camera."""
    with _transforms_lock:
        tfm = _video_transforms.get(camera_index, {"rotation": 0, "flip_h": False, "flip_v": False})
    return jsonify(tfm)

@app.route("/api/video_transform/<int:camera_index>", methods=["POST"])
def api_set_transform(camera_index):
    """Set video transform (rotation, flip) for a camera."""
    data = request.get_json()
    rotation = int(data.get("rotation", 0))
    if rotation not in (0, 90, 180, 270):
        rotation = 0
    flip_h = bool(data.get("flip_h", False))
    flip_v = bool(data.get("flip_v", False))
    with _transforms_lock:
        _video_transforms[camera_index] = {
            "rotation": rotation,
            "flip_h": flip_h,
            "flip_v": flip_v,
        }
    # Reset W/H so generate_frames picks up the new dimensions
    with streams_lock:
        stream = active_streams.get(camera_index)
        if stream:
            stream["state"].frame_size = [0, 0]
    return jsonify({"success": True, **_video_transforms[camera_index]})

@app.route("/api/logs")
def api_logs():
    """Return CSV log data."""
    try:
        rows = []
        with open("utils/data/logs/counting_data.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return jsonify(rows)
    except FileNotFoundError:
        return jsonify([])

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting People Counter Web App on http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
