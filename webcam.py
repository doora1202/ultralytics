import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./runs/detect/train/weights/best.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Obtain various properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'xvid')  
out_anno = cv2.VideoWriter('output_anno.avi', fourcc, fps, (frame_width, frame_height))#Annotation available
out_notanno = cv2.VideoWriter('output_notanno.avi', fourcc, fps, (frame_width, frame_height))#No annotation

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,max_det=1)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Get coordinates of bbox
        bounding_boxes = results[0].boxes.xyxy

        # Save bbox coordinates to file
        with open('bounding_boxes.txt', mode='a') as f:
            for box in bounding_boxes:
                f.write(str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + '\n')
        
        #Save frame to video
        out_anno.write(annotated_frame)
        out_notanno.write(frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()