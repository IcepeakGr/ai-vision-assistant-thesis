from ultralytics import YOLO  # Import YOLO model from ultralytics
import cv2  # Import OpenCV for image and video processing
import supervision as sv  # Import supervision for annotations and object detection utilities
import speech_recognition as sr  # Import speech recognition library for converting speech to text
from gtts import gTTS  # Import Google Text-to-Speech for converting text to speech
import pygame  # Import pygame for playing audio
from io import BytesIO  # Import BytesIO for handling binary streams in memory
import pyttsx3  # Import pyttsx3 for text-to-speech conversion (offline)
import threading  # Import threading for running multiple threads concurrently
import sys  # Import sys for accessing system-specific parameters and functions
import time # Import time to calculate the processing time of the program
import matplotlib.pyplot as plt # Import matplotlib to design the graphs
import pandas as pd # Import the pandas library to handle the data
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from sklearn.metrics import precision_score


# Define the Region of Interest (ROI)
# Example: ROI as a rectangle (x, y, width, height)
roi_x, roi_y, roi_w, roi_h = 100, 100, 150, 150  # Coordinates and dimensions of the ROI

model = YOLO("yolov8n.pt")  # Load the YOLOv8n model with pre-trained weights

#metrics = model.val()
#print(metrics)


# Heights of Yolov8 class objects in real life
# to calculate the distance from camera using triangle similarity
real_heights = [  # Real-world average heights of objects corresponding to YOLO class IDs
    1.75,  # person (average human height)
    1.0,  # bicycle
    1.5,  # car
    1.2,  # motorcycle
    3.0,  # airplane (height at fuselage)
    3.2,  # bus
    3.0,  # train
    3.5,  # truck
    2.5,  # boat (small boat)
    2.0,  # traffic light
    1.0,  # fire hydrant
    2.0,  # stop sign
    1.5,  # parking meter
    0.8,  # bench
    0.3,  # bird
    0.25,  # cat
    0.5,  # dog
    1.6,  # horse
    1.0,  # sheep
    1.5,  # cow
    4.0,  # elephant
    2.0,  # bear
    1.5,  # zebra
    5.0,  # giraffe
    0.4,  # backpack
    1.0,  # umbrella
    0.3,  # handbag
    0.6,  # tie
    0.7,  # suitcase
    0.3,  # frisbee
    1.7,  # skis
    1.7,  # snowboard
    0.23,  # sports ball
    0.7,  # kite
    1.0,  # baseball bat
    0.28,  # baseball glove
    0.15,  # skateboard
    1.8,  # surfboard
    1.1,  # tennis racket
    0.3,  # bottle
    0.15,  # wine glass
    0.15,  # cup
    0.2,  # fork
    0.2,  # knife
    0.2,  # spoon
    0.05,  # bowl
    0.2,  # banana
    0.1,  # apple
    0.15,  # sandwich
    0.1,  # orange
    0.15,  # broccoli
    0.15,  # carrot
    0.2,  # hot dog
    0.05,  # pizza
    0.1,  # donut
    0.15,  # cake
    1.0,  # chair
    0.8,  # couch
    0.6,  # potted plant
    0.7,  # bed
    0.75,  # dining table
    0.5,  # toilet
    0.6,  # TV
    0.4,  # laptop
    0.1,  # mouse
    0.05,  # remote
    0.05,  # keyboard
    0.15,  # cell phone
    0.3,  # microwave
    0.5,  # oven
    0.3,  # toaster
    0.3,  # sink
    1.8,  # refrigerator
    0.3,  # book
    0.3,  # clock
    0.3,  # vase
    0.1,  # scissors
    0.3,  # teddy bear
    0.2,  # hair drier
    0.2  # toothbrush
]

#tts_engine = pyttsx3.init()  # Initialize the pyttsx3 text-to-speech engine

Threadlock = threading.Lock()  # Create a threading lock to prevent race conditions

detected_names_Set = set()  # Create a set to store detected object names to be announced

announced_thread_running = False  # Flag to track if the announcement thread is running

stop_flag = False  # Flag to stop processes if needed

focalLength = 220  # Pre-determined focal length of the camera in pixels (for distance calculation)

box_annotator = sv.BoundingBoxAnnotator(thickness=2)  # Create a bounding box annotator with specified thickness
label_annotator = sv.LabelAnnotator()  # Create a label annotator for adding labels to detected objects
names = model.names  # Get the list of object class names from the model


# Add a global stop flag for the TTS thread
stop_tts_flag = False


# This is a function that announces the detected class names
# Because the Multi-threading was announcing only 1 class name
# The detected class name is being added in a Set to be stored
# and once it is being announced by the text_to_speech() functions, it pops.

def announce_detected_names():
    global announced_thread_running  # Indicate that the function modifies the global variable

    announced_thread_running = True  # Set the thread running flag to True

    while detected_names_Set and not stop_tts_flag:  # Continue as long as there are names in the set
        with Threadlock:  # Acquire the thread lock to ensure thread-safe operations
            if detected_names_Set:  # Check if there are still names to announce
                detected_class_name = detected_names_Set.pop()  # Pop a name from the set
                text_to_speech(detected_class_name)
    announced_thread_running = False  # Set the thread running flag to False when done


def speech_to_text():
    recognizer = sr.Recognizer()  # Create a Recognizer object for speech recognition
    with sr.Microphone() as source:  # Use the microphone as the audio source
        print("Listening...")  # Notify that the program is listening for speech
        recognizer.adjust_for_ambient_noise(source)  # Adjust the recognizer for ambient noise
        # source, timeout, phrase time limit
        audio = recognizer.listen(source, 10, 3)  # Listen to the audio input with a timeout and phrase time limit

    try:
        text = recognizer.recognize_google(audio)  # Try to recognize the speech using Google's API
        print("You said:", text)  # Print the recognized text
        return text  # Return the recognized text
    except sr.UnknownValueError:  # Handle the case where the speech was not understood
        print("Could not understand audio")  # Notify that the audio was not understood
        return ""  # Return an empty string
    except sr.RequestError as e:  # Handle API request errors
        print("Could not request results; {0}".format(e))  # Print the error message
        return ""  # Return an empty string


def detect_stop_speech():
    recognizer = sr.Recognizer()  # Create a Recognizer object for speech recognition
    microphone = sr.Microphone()  # Create a Microphone object to use as the audio source

    while not stop_flag:  # Continue the loop until the stop_flag is set to True
        with microphone as source:  # Use the microphone as the audio source
            with Threadlock:  # Acquire the thread lock to prevent race conditions
                recognizer.adjust_for_ambient_noise(source)  # Adjust the recognizer for ambient noise
                print("Listening for 'stop' command...")  # Notify that the program is listening for the "stop" command
                try:
                    audio = recognizer.listen(source)  # Listen to the audio input
                    command = recognizer.recognize_google(audio)  # Try to recognize the speech using Google's API
                    if command.lower() == "stop":  # Check if the recognized command is "stop"
                        stop_flag = True  # Set the stop_flag to True to stop the loop
                        print("Stop command detected")  # Print a message indicating the stop command was detected
                        break  # Exit the loop
                except sr.UnknownValueError:  # Handle the case where the speech was not understood
                    print("Could not understand the audio")  # Notify that the audio was not understood
                except sr.RequestError as e:  # Handle API request errors
                    print(f"Could not request results; {e}")  # Print the error message
                except sr.WaitTimeoutError:  # Handle the case where listening timed out
                    print("Listening timed out")  # Notify that the listening timed out


def text_to_speech(text):
    # Convert the given text to speech and play it using pygame

    # tts_lock is commented out, could be used for thread safety if uncommented
    tts = gTTS(text, lang='en')  # Convert the given text to speech in English using gTTS
    audio_file = BytesIO()  # Create an in-memory binary stream to hold the audio data
    tts.write_to_fp(audio_file)  # Write the generated speech audio to the in-memory stream
    audio_file.seek(0)  # Reset the stream position to the beginning

    pygame.init()  # Initialize pygame

    pygame.mixer.music.load(audio_file)  # Load the audio data from the in-memory stream into pygame's mixer
    pygame.mixer.music.play()  # Play the audio
    while pygame.mixer.music.get_busy():  # Wait until the audio playback is complete
        pygame.time.Clock().tick(10)  # Slow down the loop to avoid excessive CPU usage


def calculate_distance(H_real, focal, H_image):
    # Calculate the distance of an object from the camera using triangle similarity.

    # H_real (float): Real height of the object in real-world units (e.g., meters).
    # focal (float): Focal length of the camera in pixels.
    # H_image (float): Height of the object in the image in pixels.

    # Returns:
    # float: Distance from the camera to the object in the same units as H_real.

    return (H_real * focal) / H_image  # Calculate and return the distance based on the provided parameters

# This is where the magic happens
def main():
    global stop_tts_flag # Access the global stop flag for TTS

    # AI Welcomes the user
    text_to_speech("Hello! My name is Chloe. How Can I help you today?")  # Welcome message using text-to-speech

    # Speech To Text
    while True:  # Start an infinite loop to continuously listen for commands
        speech_text = speech_to_text()  # Convert the user's speech to text

        if "walk" in speech_text:  # If the user says "walk"
            walking_mode()  # Activate the walking mode
        elif "find" in speech_text:  # If the user says "find"
            lower_class_names = [name.lower() for name in model.names.values()]  # Convert all object class names from the YOLO model to lowercase
            found_class = False  # Flag to track if a matching object class is found
            words = speech_text.lower().split()  # Split the user's speech into individual words and convert them to lowercase
            print("words: ", words)  # Debugging: Print the list of words from the speech for reference

            for i in range(len(words)):  # Iterate over each word in the command
                # Check if the current word matches any single-word class name in the YOLO model
                if words[i] in lower_class_names:
                    temp = words[i]  # Store the matched word (class name) in the 'temp' variable
                    found_class = True  # Set the 'found_class' flag to True since we found a match
                    updated_search_mode(temp)  # Activate the search mode for the detected object class
                    break  # Exit the loop once a match is found

                # Check for compound-word matches (two adjacent words)
                if i < len(words) - 1:  # Ensure there are at least two words left to check
                    combined_words = f"{words[i]} {words[i + 1]}"  # Combine the current word with the next word
                    if combined_words in lower_class_names:  # Check if the combined two-word phrase matches a class name
                        temp = combined_words  # Store the combined class name in 'temp'
                        found_class = True  # Set the 'found_class' flag to True since we found a match
                        updated_search_mode(temp)  # Activate the search mode for the detected object class
                        break  # Exit the loop once a match is found



            if not found_class:  # If no matching object class was found
                text_to_speech("Invalid Searching Argument")  # Inform the user of an invalid search argument
                break  # Exit the loop

        elif "terminate" in speech_text:  # If the user says "terminate"
            text_to_speech("Shutting Down. Bye Bye")  # Announce shutdown using text-to-speech
            exit()  # Terminate the program
        else:
            text_to_speech("Invalid Mode. Please try again.")  # Inform the user of an invalid mode command




def walking_mode():
    processing_times = [] # List that holds the processing times of each frame
    cpu_usage_list, memory_usage_list, disk_usage_list = [], [], []  # Lists to hold hardware usage data
    text_to_speech("Entering Walking Mode")  # Announce that the walking mode is being entered

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize video capture from the default camera

    average_processing_time = 0
    frame_count = 1
    while True:  # Start an infinite loop to continuously process frames from the camera

        start_time = time.time()

        ret, frame = cap.read()  # Capture a frame from the camera

        # if frame is read correctly ret is True
        while ret == False:  # If frame capture fails (ret is False)
            print("Can't receive frame. Retrying ...")  # Print a message indicating the failure
            cap.release()  # Release the camera resource
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Reinitialize the video capture
            ret, frame = cap.read()  # Attempt to capture the frame again

        if not ret or frame is None:  # If ret is False or the frame is None (not captured correctly)
            print("Debug: Failed to capture frame, ret:", ret)  # Print a debug message
            continue  # Skip to the next iteration of the loop

        if frame is None or frame.size == 0:  # Check if the frame is invalid (None or empty)
            print("Invalid frame detected, skipping...")  # Print a message indicating an invalid frame
            continue  # Skip to the next iteration of the loop


        model_track = model.predict(source=frame)  # Use the model to make predictions on the current frame



        result = model(frame, iou=0.3)[0]  # Get the detection results with an Intersection over Union (IoU) threshold of 0.4

        detections = sv.Detections.from_ultralytics(result)  # Convert the detection results into a usable format


        height_in_pixels, class_ids = HandleBoundingBoxesHeight(model_track)  # Calculate the height of the detected objects in pixels

        labels = []  # Initialize an empty list to store detection labels
        # Iterate over each detected object and calculate its distance
        for i, height in enumerate(height_in_pixels):
            objectClassID = class_ids[i]  # Use the correct class ID for each object

            if objectClassID is not None and isinstance(objectClassID, int) and 0 <= objectClassID < len(real_heights):
                # Calculate distance for the object using its real height and bounding box height
                distance = calculate_distance(real_heights[objectClassID], focalLength, height)
                steps = distance / 0.76  # Convert distance to steps
                detection_message = f"WM: {names[objectClassID]}, {distance:.1f} m, {steps:.1f} steps"  # Create a detection message
                labels.append(detection_message)  # Add the detection message to the labels list
                detected_names_Set.add(
                    f"{names[objectClassID]} detected at {distance:.1f} m or {steps:.1f} steps")  # Add to set

        if detected_names_Set and not announced_thread_running:  # If there are detected names and the announcement thread is not running
            tts_thread = threading.Thread(target=announce_detected_names, daemon=True)  # Create a new thread to announce the detected names
            tts_thread.start()  # Start the thread
        else:
            detected_names_Set.clear()  # Clear the set of detected names if no announcement is made

        frame = box_annotator.annotate(scene=frame, detections=detections)  # Annotate the frame with bounding boxes
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)  # Annotate the frame with detection labels



        cv2.imshow("Camera", frame)  # Display the annotated frame in a window

        # We create the thread so that the Voice-over can run asynchronously
        tts_thread = threading.Thread(target=announce_detected_names, daemon=True)  # Create a thread to handle voice announcements
        tts_thread.start()  # Start the thread

        if (cv2.waitKey(1) & 0xFF == ord('x')):  # Check if the 'x' key is pressed
            stop_tts_flag = True # Set the stop flag to True to stop the TTS thread
            break  # Exit the program

        print("Model Tracking", model_track)  # Print the model tracking results for debugging
        print("Specific detections = ", detections.class_id, "$$")  # Print the specific detections for debugging
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        print(f"Current frame: {frame_count}")
        print(f"Time to process frame: {processing_time} seconds")


        frame_count += 1





    # Calculate precision, recall, F1 score and confusion matrix after exiting loop
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F1 Score: {f1:.2f}')

    # # Confusion matrix for the classes detected
    # known_labels = list(range(len(model.names)))
    # cm = confusion_matrix(y_true, y_pred, labels=known_labels)
    #
    # plt.figure(figsize=(10, 7))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    #
    # tick_marks = np.arange(len(model.names))
    # plt.xticks(tick_marks, model.names.values(), rotation=45, ha="right")
    # plt.yticks(tick_marks, model.names.values())
    #
    # plt.show()


    # The code below was used to extract the model statistics and has been commented out since it serves us no further
    #
    # # Calculate the average processing time
    # average_processing_time = sum(processing_times) / len(processing_times)
    #
    #
    # # Save processing times and frame numbers in a DataFrame
    # df = pd.DataFrame({
    #     'Frame': range(1, len(processing_times) + 1),
    #     'Processing Time (s)': processing_times
    # })
    #
    # # Save the DataFrame to a CSV file
    # df.to_csv('frame_processing_times.csv', index=False)
    #
    # # Plot the frame processing times
    # plt.plot(df['Frame'], df['Processing Time (s)'], label='Processing Time')
    #
    # # Plot the average processing time as a horizontal line
    # plt.axhline(y=average_processing_time, color='r', linestyle='--',
    #             label=f'Average Time: {average_processing_time:.2f} s')
    #
    # plt.xlabel('Frame')
    # plt.ylabel('Processing Time (s)')
    # plt.title('WM: Processing Time per Frame with Average Time')
    # plt.legend()
    # plt.show()


    # fps = 1 / average_processing_time
    # print(f"FPS: {fps}")

    exit()




def updated_search_mode(temp):
    processing_times = []

    text_to_speech("Entering Search Mode")  # Announce that the search mode is being entered

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize video capture from the default camera


    if temp in list(model.names.values()):  # Check if the search term is a valid class in the YOLO model
        text_to_speech(f"Searching for {temp}")  # Announce what object is being searched for

        # Detect the YOLOv8 class ID by using the index of the list
        if list(model.names.values()).index(temp) >= 0 and (list(model.names.values()).index(temp)) <= len(list(model.names.values())):
            # Ensure the index is valid
            print("Found Class ", temp, list(model.names.values()).index(temp))  # Print the found class and its index
            class_index = list(model.names.values()).index(temp)  # Store the class index for use in detection
    else:
        print("Parameter not found in the dataset. Current model needs to be trained further.")  # Inform the user if the class is not found
        text_to_speech("Invalid Searching Argument")  # Announce that the search term is invalid
        exit()  # Exit the program


    average_processing_time = 0
    frame_count = 1

    while True:  # Start an infinite loop to continuously process frames from the camera

        start_time = time.time()

        ret, frame = cap.read()  # Capture a frame from the camera


        pred = model.predict(source=frame, classes=class_index)  # Predict using YOLO model for the specific class

        heights_in_pixels, class_ids = HandleBoundingBoxesHeight(pred) # Calculate the height of the detected objects in pixels
        labels = []  # Initialize an empty list to store detection labels

        # Run YOLOv8 on the frame with an Intersection over Union (IoU) threshold of 0.4
        results = model(frame, iou=0.4)[0]

        detections = sv.Detections.from_ultralytics(results)  # Convert the detection results into a usable format



        # Iterate over each detected object and calculate its distance
        for i, height_in_pixels in enumerate(heights_in_pixels):
            objectClassID = class_ids[i]  # Get the correct class ID for each object



            if objectClassID == class_index:  # Only process the target class
                if objectClassID is not None and 0 <= objectClassID < len(real_heights):
                    # Calculate the distance for the object using its real height and bounding box height
                    distance = calculate_distance(real_heights[objectClassID], focalLength, height_in_pixels)
                    steps = distance / 0.76  # Convert distance to steps
                    detection_message = f"SM: {temp}, {distance:.1f} m, {steps:.1f} steps"  # Create a detection message
                    labels.append(detection_message)  # Add the detection message to the labels list
                    detected_names_Set.add(f"{temp} detected at {distance:.1f} m or {steps:.1f} steps")  # Add to set

        # Check if there are detected names and the announcement thread is not running
        if detected_names_Set and not announced_thread_running:
            # Add the detection message to the set of detected names
            detected_names_Set.add(f"{temp} detected at {distance:.1f} m or {steps: .1f} steps")
            tts_thread = threading.Thread(target=announce_detected_names, daemon=True)  # Create a thread to announce detected names
            tts_thread.start()  # Start the thread
        else:
            detected_names_Set.clear()  # Clear the set of detected names if no announcement is made

        frame = box_annotator.annotate(scene=frame, detections=detections)  # Annotate the frame with bounding boxes
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)  # Annotate the frame with detection labels
        cv2.imshow("Camera", frame)  # Display the annotated frame in a window

        # Exit the loop and program if the 'x' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('x'):
            stop_tts_flag = True # Set the stop flag to True to stop the TTS thread
            break  # Exit the program

        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)  # Save the processing time
        print(f"Current frame: {frame_count}")
        print(f"Time to process frame: {processing_time} seconds")

        #average_processing_time += processing_time
        frame_count += 1

    # known_labels = list(range(len(model.names)))  # Λίστα με όλους τους πιθανούς class IDs
    # cm = confusion_matrix(y_true, y_pred, labels=known_labels)
    #
    # plt.figure(figsize=(10, 7))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    #
    # tick_marks = np.arange(len(model.names))
    # plt.xticks(tick_marks, model.names.values(), rotation=45, ha="right")
    # plt.yticks(tick_marks, model.names.values())
    #
    # plt.show()
    #
    # # Save processing times and frame numbers in a DataFrame
    # df = pd.DataFrame({
    #     'Frame': range(1, len(processing_times) + 1),
    #     'Processing Time (s)': processing_times
    # })
    #
    #
    # # The code below was used to extract the model statistics and has been commented out since it serves us no further
    #
    # # # Calculate the average processing time
    # average_processing_time = sum(processing_times) / len(processing_times)
    #
    # # Save the DataFrame to a CSV file
    # df.to_csv('frame_processing_times.csv', index=False)
    #
    # # Plot the frame processing times
    # plt.plot(df['Frame'], df['Processing Time (s)'], label='Processing Time')
    #
    # # Plot the average processing time as a horizontal line
    # plt.axhline(y=average_processing_time, color='r', linestyle='--',
    #             label=f'Average Time: {average_processing_time:.2f} s')
    #
    # plt.xlabel('Frame')
    # plt.ylabel('Processing Time (s)')
    # plt.title('SM: Processing Time per Frame with Average Time')
    # plt.legend()
    # plt.show()
    #
    # fps = 1 / average_processing_time
    # print(f"FPS: {fps}")

    exit()



def HandleBoundingBoxesHeight(predictions):
    names = model.names  # Get the list of class names from the YOLO model
    Bounding_Box_heights_in_pixels = []  # List to store the heights of all detected objects
    class_ids = []  # List to store the class IDs for all detected objects

    for r in predictions:  # Iterate through the predictions

        if stop_flag:  # Check if the stop flag is set, and if so, break the loop
            break

        for box in r.boxes:  # Iterate through the bounding boxes in each prediction
            detected_class_id = int(box.cls)  # Get the class ID for the detected object
            detected_name = names[detected_class_id]  # Get the class name using the class ID
            detected_names_Set.add(detected_name)  # Add the detected class name to the set

            # Store the class ID and name for later use
            class_ids.append(detected_class_id)
            detected_names_Set.add(detected_name)  # Add the detected class name to the set
            print(f"Detected: {detected_name}")  # Print the detected class name

            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box

            # Calculate height in pixels by subtracting the top y-coordinate from the bottom y-coordinate
            height_in_pixels = y2 - y1
            Bounding_Box_heights_in_pixels.append(height_in_pixels)  # Store the height for this object

            # Debugging: Print the bounding box height
            print(f"Bounding Box Height (in pixels): {height_in_pixels}")

    return Bounding_Box_heights_in_pixels, class_ids  # Return the list of all bounding box heights and corresponding class IDs


def getObjectClassID(predictions):
    names = model.names  # Get the list of class names from the YOLO model

    for r in predictions:  # Iterate through the predictions
        for box in r.boxes:  # Iterate through the bounding boxes in the prediction
            class_id = int(box.cls)  # Get the class ID from the bounding box
            return class_id  # Return the class ID


if __name__ == "__main__":
    main()  # Call the main function to start the program
