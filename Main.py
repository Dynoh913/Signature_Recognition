import tkinter as tk
from PIL import ImageTk, Image, ImageOps
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 29)
    binary = cv2.bitwise_not(binary)

    kernel_close = np.ones((14, 14), np.uint8)
    closed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_open)
    contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]
    x_min, y_min = np.inf, np.inf
    x_max, y_max = 0, 0
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    padding = 50
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
    ROI = image[y_min:y_max, x_min:x_max]
    resized_ROI = cv2.resize(ROI, (224, 224))
    resized_ROI = (resized_ROI / 127.5) - 1
    resized_ROI = (resized_ROI * 127.5 + 127.5).astype(np.uint8)
    return resized_ROI


def preprocess_image_camera(image):
    # Resize the image to 224x224 and convert to RGB
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image = (image / 127.5) - 1
    return image

def predict_signature(image):
    # Reshape image for model input
    image = image.reshape(1, 224, 224, 3)

    # Predict using the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def open_roi_image():
    file_path = filedialog.askopenfilename()

    # Let the user select a region of interest (ROI)
    roi = cv2.selectROI("Select ROI", cv2.imread(file_path), fromCenter=False)

    # Close the ROI selection window
    cv2.destroyWindow("Select ROI")

    # Read the image
    image = cv2.imread(file_path)

    # Extract ROI coordinates and center
    x, y, w, h = roi
    cx, cy = x + w // 2, y + h // 2

    # Define the size of the fixed ROI
    fixed_size = 224
    half_size = fixed_size // 2

    # Calculate the new ROI coordinates ensuring it is within image boundaries
    x_min = max(cx - half_size, 0)
    y_min = max(cy - half_size, 0)
    x_max = min(cx + half_size, image.shape[1])
    y_max = min(cy + half_size, image.shape[0])

    # Ensure the ROI is exactly 224x224 by adjusting if it is near the image boundaries
    if x_max - x_min != fixed_size:
        if x_min == 0:
            x_max = fixed_size
        else:
            x_min = x_max - fixed_size

    if y_max - y_min != fixed_size:
        if y_min == 0:
            y_max = fixed_size
        else:
            y_min = y_max - fixed_size

    # Crop the ROI from the image
    ROI = image[y_min:y_max, x_min:x_max]

    # Convert ROI to RGB and resize using PIL
    roi_image = Image.fromarray(cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)).convert("RGB")
    roi_image = ImageOps.fit(roi_image, (fixed_size, fixed_size), Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(roi_image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Display the processed image in a Tkinter panel (assuming you have a Tkinter panel setup)
    tk_image = ImageTk.PhotoImage(roi_image)
    panel.config(image=tk_image)
    panel.image = tk_image

    class_name, confidence_score = predict_signature(data)

    # Print prediction and confidence score
    result_label.config(text="Signature of: " + f'{class_name[2:]}{str(np.round(confidence_score * 100))[:-2]}%')



def open_full_image():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    file_path = filedialog.askopenfilename()
    imgage = cv2.imread(file_path)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(file_path).convert("RGB")

    # Find contours on extracted mask, combine boxes, and extract ROI
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    image = ImageTk.PhotoImage(image)
    panel.config(image=image)
    panel.image = image
    
    class_name, confidence_score = predict_signature(data)

    # Print prediction and confidence score
    result_label.config(text="Signature of: " + f'{class_name[2:]}{str(np.round(confidence_score * 100))[:-2]}%')


def open_camera():
    def process_frame(frame):
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        binary = cv2.bitwise_not(binary)
        kernel_close = np.ones((9, 9), np.uint8)
        edges = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if area > 100 and h < 2 * w:
                padding_size = 30
                x_new = max(0, x - padding_size)
                y_new = max(0, y - padding_size)
                w_new = w + 2 * padding_size
                h_new = h + 2 * padding_size
                signature_region = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                processed_signature = preprocess_image_camera(signature_region)
                class_name, confidence_score = predict_signature(processed_signature)
                class_name = class_name.rstrip('\n')
                cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 255, 0), 2)
                cv2.putText(frame,"Signature of: " + f'{class_name[2:]}: {str(np.round(confidence_score * 100))[:-2]}%',
                            (x_new, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, edges
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        processed_frame, processed_image = process_frame(frame)
        cv2.imshow("Video", processed_frame)
        # cv2.imshow("Processed Image", processed_image)  # Hiển thị processed_image

        if cv2.waitKey(1) == 27:  # Nhấn 'esc' để thoát
            break

    camera.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.geometry("800x600")

# Create labels and buttons
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

output_panel = tk.Label(root)
output_panel.pack(padx=10, pady=10)

result_label = tk.Label(root, text='')
result_label.pack(padx=10, pady=10)

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

button1 = tk.Button(frame, text='Sử dụng ảnh cắt', command=open_roi_image, bg='blue', fg='white', font=('Arial', 14))
button1.pack(side='left', padx=10)

button2 = tk.Button(frame, text='Sử dụng ảnh đầy đủ', command=open_full_image, bg='green', fg='white', font=('Arial', 14))
button2.pack(side='left', padx=10)

button3 = tk.Button(root, text='Sử dụng Camera', command=open_camera, bg='red', fg='white', font=('Arial', 14))
button3.pack(padx=10, pady=10)

root.mainloop()






