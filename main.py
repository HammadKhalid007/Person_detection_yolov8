import cv2
import random
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("12.mp4")

desired_width = 640
desired_height = 480

bounding_boxes = []
object_ids = []
colors = {}
selected_object_id = None
timer_start = None
trackers = {}
last_seen = {}
 # seconds to wait before removing the tracker
timeout = 10 

def generate_unique_color(exclude_colors):
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color[0] > 50 and (color[0] < 150 or color[1] < 150 or color[2] < 150):
            continue
        if color not in exclude_colors:
            return color

def draw_boxes(frame, boxes, ids):
    for i, box in enumerate(boxes):
        if box is not None:
            (x1, y1, x2, y2) = box
            object_id = ids[i]
            color = colors.get(object_id, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if object_id == selected_object_id:
                elapsed_time = int(time.time() - timer_start)
                cv2.putText(frame, f'Timer: {elapsed_time}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def handle_mouse_click(event, x, y, flags, param):
    global selected_object_id, timer_start
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (box, object_id) in enumerate(zip(bounding_boxes, object_ids)):
            (x1, y1, x2, y2) = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                if selected_object_id is not None:
                    colors[selected_object_id] = generate_unique_color(colors.values())
                selected_object_id = object_id
                colors[object_id] = (0, 0, 255)
                timer_start = time.time()
                break

cv2.namedWindow("Video Stream")
cv2.setMouseCallback("Video Stream", handle_mouse_click)