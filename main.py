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