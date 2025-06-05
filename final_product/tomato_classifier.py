import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hackathon/tomato_final_model.keras', compile=False)

# Image preprocessing function
def preprocess_image(frame):
    # Resize the image to match model's expected input size
    resized = cv2.resize(frame, (224, 224))
    # Convert to RGB (model was trained on RGB images)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    normalized = rgb / 255.0
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

# ---------------------- Main Window ---------------------- #
root = tk.Tk()
root.title("🍅 Tomato Quality Classifier")
root.geometry("900x750")
root.configure(bg="#fff7ed")  # Light warm background

# ---------------------- Fonts & Colors ---------------------- #
TITLE_FONT = ("Verdana", 28, "bold")
SUBTITLE_FONT = ("Calibri", 16)
BUTTON_FONT = ("Helvetica", 14, "bold")
RESULT_FONT = ("Helvetica", 18, "bold")

TITLE_COLOR = "#b82e1f"       # Tomato red
TEXT_COLOR = "#3e3e3e"
BUTTON_COLOR = "#2e8b57"      # Fresh green
BUTTON_HOVER = "#3cb371"
STATUS_COLOR = "#666666"

# ---------------------- Title & Subtitle ---------------------- #
title = tk.Label(
    root,
    text="🍅 Tomato Quality Classifier",
    font=TITLE_FONT,
    bg="#fff7ed",
    fg=TITLE_COLOR
)
title.pack(pady=(30, 5))

subtitle = tk.Label(
    root,
    text="Detect Tomato Quality with AI",
    font=SUBTITLE_FONT,
    bg="#fff7ed",
    fg=TEXT_COLOR
)
subtitle.pack(pady=(0, 20))

# ---------------------- Video Frame Placeholder ---------------------- #
video_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
video_label = tk.Label(video_frame, bg="#ffffff", width=720, height=480)
video_label.pack()
video_shown = False

# ---------------------- Status and Result Labels ---------------------- #
status_frame = tk.Frame(root, bg="#fff7ed")
status_frame.pack(fill="x", pady=(10, 5))

status_label = tk.Label(
    status_frame,
    text="Camera: Off (Press 'S' to start, 'Q' to stop, 'C' to capture)",
    font=("Helvetica", 12),
    bg="#fff7ed",
    fg=STATUS_COLOR
)
status_label.pack(side="left")

result_label = tk.Label(
    root,
    text="",
    font=RESULT_FONT,
    bg="#fff7ed",
    fg=TEXT_COLOR
)
result_label.pack(pady=10)

# ---------------------- Webcam Function ---------------------- #
cap = None
is_camera_running = False
current_frame = None

def start_camera():
    global cap, video_shown, is_camera_running
    if not video_shown:
        video_frame.pack(pady=20)
        video_shown = True

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        is_camera_running = True
        status_label.config(text="Camera: On (Press 'Q' to stop, 'C' to capture)", fg=BUTTON_COLOR)

    def update_frame():
        global current_frame
        if cap and cap.isOpened() and is_camera_running:
            ret, frame = cap.read()
            if ret:
                current_frame = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
            if is_camera_running:
                video_label.after(10, update_frame)

    update_frame()

def stop_camera():
    global is_camera_running, cap
    is_camera_running = False
    if cap is not None:
        cap.release()
    status_label.config(text="Camera: Off (Press 'S' to start, 'Q' to stop, 'C' to capture)", fg=STATUS_COLOR)
    result_label.config(text="")

def capture_and_analyze():
    global current_frame
    if current_frame is not None and is_camera_running:
        try:
            # Show capturing status
            status_label.config(text="📸 Capturing and analyzing...", fg=BUTTON_COLOR)
            root.update()

            # Process the captured frame
            processed_frame = preprocess_image(current_frame)
            prediction = model.predict(processed_frame, verbose=0)[0]
            
            # Get the predicted class from 4 categories
            class_names = ["Ripe", "Unripe", "Rotten", "Semi-ripe"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Set color based on category
            if predicted_class == "Ripe":
                result_color = "#b82e1f"  # Tomato red
            elif predicted_class == "Unripe":
                result_color = "#2e8b57"  # Fresh green
            elif predicted_class == "Rotten":
                result_color = "#8b0000"  # Dark red
            else:  # Semi-ripe
                result_color = "#ff8c00"  # Dark orange
            
            # Update result label
            result_text = f"Prediction: {predicted_class} (Confidence: {confidence:.2%})"
            result_label.config(text=result_text, fg=result_color)
            
            # Save the captured image
            timestamp = cv2.getTickCount()
            filename = f"captured_tomato_{timestamp}.jpg"
            
            # Draw prediction on the frame
            frame_with_text = current_frame.copy()
            cv2.putText(
                frame_with_text,
                f"{predicted_class}: {confidence:.2%}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Save both original and predicted images
            cv2.imwrite(filename, current_frame)
            cv2.imwrite(f"predicted_{filename}", frame_with_text)
            
            # Show success message
            status_label.config(text=f"✓ Captured! Saved as {filename} (Press 'C' to capture again)", fg=BUTTON_COLOR)
            
        except Exception as e:
            status_label.config(text=f"Error during capture: {str(e)}", fg="#8b0000")
    else:
        status_label.config(text="Cannot capture: Camera not running or no frame available", fg="#8b0000")

# ---------------------- Button Styles ---------------------- #
def create_button(parent, text, command, color, hover_color):
    btn = tk.Button(
        parent,
        text=text,
        font=BUTTON_FONT,
        bg=color,
        fg="white",
        padx=25,
        pady=12,
        relief="flat",
        command=command,
        cursor="hand2"
    )
    
    def on_enter(e):
        btn.config(bg=hover_color)
    
    def on_leave(e):
        btn.config(bg=color)
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

# ---------------------- Buttons ---------------------- #
button_frame = tk.Frame(root, bg="#fff7ed")
button_frame.pack(pady=20)

start_btn = create_button(button_frame, "🎥 Start Camera", start_camera, BUTTON_COLOR, BUTTON_HOVER)
stop_btn = create_button(button_frame, "⏹ Stop Camera", stop_camera, "#8b0000", "#a52a2a")
capture_btn = create_button(button_frame, "📸 Capture & Analyze", capture_and_analyze, "#ff8c00", "#ffa500")

# Bind keyboard shortcuts
root.bind('s', lambda e: start_camera())
root.bind('q', lambda e: stop_camera())
root.bind('c', lambda e: capture_and_analyze())

# ---------------------- Run the App ---------------------- #
root.mainloop()
