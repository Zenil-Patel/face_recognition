from deepface import DeepFace
import cv2
import os
import shutil

# Step 1: Capture 5 pictures with different angles from the webcam, with mirrored copies
def capture_images_with_mirroring(num_images=5):
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam. Please check your camera settings.")
        return []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    captured_images = []
    prompts = [
        "Look straight ahead", 
        "Tilt your head slightly to the left", 
        "Tilt your head slightly to the right", 
        "Look slightly up", 
        "Look slightly down"
    ]

    # Define the box dimensions where the user should place their face
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    box_width, box_height = 250, 250
    box_x = (frame_width - box_width) // 2
    box_y = (frame_height - box_height) // 2

    for i in range(num_images):
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break

            # Mirror the frame horizontally
            mirrored_frame = cv2.flip(frame, 1)

            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw the rectangle where the user should place their face
            cv2.rectangle(mirrored_frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)
            cv2.putText(mirrored_frame, f"Place your face inside the box", (box_x - 20, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            face_inside_box = False  # Check if face is inside the box

            # Loop through detected faces and check if any face is inside the box
            for (x, y, w, h) in faces:
                if (box_x <= x <= box_x + box_width - w) and (box_y <= y <= box_y + box_height - h):
                    face_inside_box = True
                    break

            if len(faces) > 1:
                cv2.putText(mirrored_frame, "Multiple faces detected! Clear the background.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) == 0 or not face_inside_box:
                cv2.putText(mirrored_frame, "No face inside the box! Please adjust your position.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(mirrored_frame, prompts[i], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(mirrored_frame, f"Captured Images: {i}/{num_images}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('Video', mirrored_frame)

            # Capture the image only when 's' key is pressed
            if face_inside_box and len(faces) == 1 and cv2.waitKey(1) & 0xFF == ord('s'):
                image_path = f"temp/captured_image_{i}.jpg"
                mirrored_image_path = f"temp/captured_image_{i}_mirrored.jpg"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                # Save the original mirrored frame
                cv2.imwrite(image_path, mirrored_frame)
                captured_images.append(image_path)

                # Save the additional mirrored version
                flipped_image = cv2.flip(mirrored_frame, 1)
                cv2.imwrite(mirrored_image_path, flipped_image)
                captured_images.append(mirrored_image_path)

                print(f"Captured image {i+1}/{num_images}")

                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

    return captured_images

# Step 2: Recognize faces using multiple models with a threshold of 0.5
def recognize_faces_from_images(captured_image_paths, stored_image_paths, match_folder="matched_images", threshold=0.5):
    os.makedirs(match_folder, exist_ok=True)  # Create folder for matched images if it doesn't exist

    models = ['VGG-Face', 'Facenet', 'OpenFace']  # List of models to test
    matched_images = set()  # Use a set to avoid duplicates

    for captured_image in captured_image_paths:
        print(f"Processing captured image: {captured_image}")

        for stored_image in stored_image_paths:
            print(f"Comparing with stored image: {stored_image}")

            # Process with multiple models
            for model in models:
                try:
                    # Verify the face match with the current model
                    result = DeepFace.verify(img1_path=captured_image, img2_path=stored_image, model_name=model, enforce_detection=False)

                    # Check if the distance is below the threshold (0.5 in this case)
                    if result['verified']:
                        print(f"Match found using {model} model for {captured_image} with {stored_image} (distance: {result['distance']})")

                        # Copy the matched image to the matched_images folder if not already copied
                        if stored_image not in matched_images:
                            filename = os.path.basename(stored_image)
                            matched_image_path = os.path.join(match_folder, filename)
                            shutil.copy(stored_image, matched_image_path)
                            matched_images.add(stored_image)  # Add to the set of matches
                        break  # No need to check with other models once matched
                except Exception as e:
                    print(f"Error processing {stored_image}: {e}")
                    continue

    # At the end, display all matched images
    print(f"Total matched images: {len(matched_images)}")
    for match in matched_images:
        print(f"Match: {match}")

if __name__ == "__main__":
    # Step 1: Capture 5 images from the mirrored webcam feed with mirrored copies
    captured_images = capture_images_with_mirroring(num_images=5)

    # Step 2: Load stored images from folder to search against
    folder_path = "path_to_folder"  # Specify your folder
    stored_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Step 3: Use captured images to search through stored images and store matches
    recognize_faces_from_images(captured_images, stored_image_paths, match_folder="matched_images", threshold=0.5)
