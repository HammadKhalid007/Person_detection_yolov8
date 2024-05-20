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

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (desired_width, desired_height))
        results = model.track(frame, persist=True)
        new_bounding_boxes = []
        new_object_ids = []

        current_time = time.time()

        for result in results:
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                if class_id == 'person':
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    new_bounding_boxes.append(cords)
                    new_object_ids.append(box.id[0].item() if box.id is not None else len(new_object_ids))
                    if box.id is not None:
                        last_seen[box.id[0].item()] = current_time

        # Update existing trackers and remove those that have timed out
        active_bounding_boxes = []
        active_object_ids = []

        for object_id in list(trackers.keys()):
            if object_id in trackers:
                tracker, start_time = trackers[object_id]
                success, bbox = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in bbox]
                    active_bounding_boxes.append([x, y, x + w, y + h])
                    active_object_ids.append(object_id)
                    last_seen[object_id] = current_time

                    # Reset last seen time if person is detected again
                    if object_id in new_object_ids:
                        last_seen[object_id] = current_time

                    # Check if tracker should be removed due to timeout
                    if current_time - start_time > timeout:
                        print(f"Stopping tracker for object_id {object_id} due to timeout.")
                        del trackers[object_id]
                else:
                    print(f"Tracker failed for object_id {object_id}.")
                    del trackers[object_id]

        # Initialize new trackers for newly detected persons
        for i, box in enumerate(new_bounding_boxes):
            object_id = new_object_ids[i]
            if object_id not in trackers:
                tracker = cv2.TrackerCSRT_create()
                bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                tracker.init(frame, bbox)
                trackers[object_id] = (tracker, current_time)
                
        bounding_boxes = active_bounding_boxes
        object_ids = active_object_ids

        for i, object_id in enumerate(object_ids):
            if object_id not in colors:
                colors[object_id] = generate_unique_color(colors.values())

        draw_boxes(frame, bounding_boxes, object_ids)
        selected_objects = [object_id for object_id in object_ids if object_id == selected_object_id]
        print(f"Selected Object IDs: {selected_objects}")

        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
