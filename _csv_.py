import csv
import cv2
from ultralytics import YOLO

def write_bbox_centers_to_csv(video_path, model, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open the video file.")
        return

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the CSV file and write the header
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Center_X', 'Center_Y'])

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection on the frame using the model
            results = model(frame, max_det=1,conf = 0.5)

            # Write the center coordinates of the bounding boxes to the CSV file
            if len(results[0].boxes.xyxy) > 0:
                for i, bbox in enumerate(results[0].boxes.xyxy.tolist()):
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    writer.writerow([frame_count, x_center, y_center])
            else:
                writer.writerow([frame_count, '', ''])

            frame_count += 1

    cap.release()

# Example usage:
video_path = "video_path"
model = YOLO('./runs/detect/train/weights/best.pt')
output_csv = "csv/output.csv"

write_bbox_centers_to_csv(video_path, model, output_csv)
