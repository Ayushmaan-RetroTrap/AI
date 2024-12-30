import cv2

# Initialize HOG descriptor and set SVM detector (HOG+SVM)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people in the image (returns bounding boxes)
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        # Draw the bounding box around the detected person
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the head (top) and feet (bottom) positions using the bounding box
        head_y = y  # The top of the bounding box (head area)
        feet_y = y + h  # The bottom of the bounding box (feet area)

        # Draw circles for the head and feet
        cv2.circle(frame_resized, (x + w // 2, head_y), 5, (0, 0, 255), -1)  # Red for head
        cv2.circle(frame_resized, (x + w // 2, feet_y), 5, (255, 0, 0), -1)  # Blue for feet

        # Annotate with text
        cv2.putText(frame_resized, 'Head', (x + w // 2, head_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame_resized, 'Feet', (x + w // 2, feet_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Body Detection', frame_resized)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
