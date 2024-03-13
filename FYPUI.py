"""import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# Declare global variables first
result_frame = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

# Function to open and analyze the file
def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        # Clear previous data
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        # Display image
        load_image(filename)

        # Analyze and display analytics
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))  # Resize to fit the canvas
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo  # Keep reference
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame  # Declare result_frame as global here
    try:
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.06, minNeighbors=3, minSize=(15,15))

        if len(faces) > 0:
            print(f"Number of faces detected: {len(faces)}")
            if result_frame:
                result_frame.destroy()  # Destroy previous result frame

            # Create a new Frame to hold the analysis results
            result_frame = tk.Frame(root, bg="#FFFFCC", bd=5)
            result_frame.pack(side="right", fill="y", padx=20, pady=20)

            for index, (x, y, w, h) in enumerate(faces, start=1):
                face_img = image[y:y+h, x:x+w]

                # Analyze the extracted face with DeepFace
                analysis_results = DeepFace.analyze(face_img, actions=['emotion', 'gender'], enforce_detection=False)
                
                # Accessing the analysis results
                emotion_info = analysis_results['dominant_emotion']
                gender_info = analysis_results['dominant_gender']

                # Display the analysis results for each face
                result_text = f"Face {index}: \nEmotion - {emotion_info}\nGender - {gender_info}"
                result_label = tk.Label(result_frame, text=result_text, wraplength=250, justify="left", bg="#FFFFCC")
                result_label.grid(row=index, column=0, padx=10, pady=5, sticky="w")

        else:
            messagebox.showinfo("Information", "No faces detected in the image.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Main window setup
root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

style = ttk.Style(root)
style.configure('TButton', font=('Arial', 10), borderwidth='1')
style.map('TButton', foreground=[('pressed', 'red'), ('active', 'blue')], background=[('pressed', '!disabled', 'black'), ('active', 'white')])

# Main layout frame
main_frame = ttk.Frame(root, padding="10 10 10 10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Image display canvas
canvas = tk.Canvas(main_frame, width=250, height=250, bg="white")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

# File open button
open_button = ttk.Button(main_frame, text="Open Image", command=open_file)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

# Analytics labels
lbl_faces_detected = ttk.Label(main_frame, text="Faces Detected: N/A", font=('Arial', 10))
lbl_faces_detected.grid(row=2, column=0, sticky=tk.W)

lbl_overall_emotion = ttk.Label(main_frame, text="Overall Emotion Score: N/A", font=('Arial', 10))
lbl_overall_emotion.grid(row=3, column=0, sticky=tk.W)

# Results display frame
result_frame = ttk.LabelFrame(main_frame, text="Detailed Analysis", padding="10 10 10 10")
result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

# Ensure resizable window
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(4, weight=1)  # Allow results frame to expand

root.mainloop()"""

#second attempt 
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

# Function to open and analyze the file
def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        # Clear previous data
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        # Display image
        load_image(filename)

        # Analyze and display analytics
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))  # Resize to fit the canvas
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo  # Keep reference
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame  # Declare result_frame as global here
    verified_faces_count = 0  # Initialize count of verified faces
    emotions = []  # List to hold emotions of all verified faces

    try:
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.06, minNeighbors=3, minSize=(15,15))

        if len(faces) > 0:
            for index, (x, y, w, h) in enumerate(faces, start=1):
                face_img = image[y:y+h, x:x+w]

                try:
                    # Analyze the extracted face with DeepFace
                    analysis_results = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    verified_faces_count += 1  # Increment count for a successfully analyzed face

                    # Accessing the analysis results
                    emotion_info = analysis_results['dominant_emotion']
                    emotions.append(emotion_info)  # Add detected emotion to the list

                    # Display the analysis results for each verified face
                    result_text = f"Face {index}: \nEmotion - {emotion_info}"
                    result_label = tk.Label(result_frame, text=result_text, wraplength=250, justify="left", bg="#f0f0f0")
                    result_label.pack(anchor='w', padx=10, pady=5)

                except Exception as deepface_error:
                    print(f"DeepFace failed to analyze face {index}: {deepface_error}")

            # Update the Faces Detected label
            lbl_faces_detected.config(text=f"Faces Detected: {verified_faces_count}")

            # Calculate the most common emotion among all verified faces
            if emotions:
                most_common_emotion = max(set(emotions), key=emotions.count)
                lbl_overall_emotion.config(text=f"Overall Emotion: {most_common_emotion}")
            else:
                lbl_overall_emotion.config(text="Overall Emotion: N/A")

            if verified_faces_count == 0:
                messagebox.showinfo("Information", "No faces verified in the image.")
        else:
            messagebox.showinfo("Information", "No faces detected in the image.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Main window setup
root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

# Main layout frame
main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(fill="both", expand=True)

# Image display canvas
canvas = tk.Canvas(main_frame, width=250, height=250, bg="white", bd=2, relief="groove")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

# File open button
open_button = tk.Button(main_frame, text="Open Image", command=open_file, bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

# Analytics labels
lbl_faces_detected = tk.Label(main_frame, text="Faces Detected: N/A", bg="#f0f0f0", font=('Arial', 10))
lbl_faces_detected.grid(row=2, column=0, sticky="w")

lbl_overall_emotion = tk.Label(main_frame, text="Overall Emotion: N/A", bg="#f0f0f0", font=('Arial', 10))
lbl_overall_emotion.grid(row=3, column=0, sticky="w")

# Results display frame
result_frame = tk.Frame(main_frame, bg="#f0f0f0")
result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

# Ensure resizable window
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(4, weight=1)  # Allow results frame to expand

root.mainloop()"""

#3RD ATTEMPT 
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace

def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        load_image(filename)
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame, lbl_faces_detected, lbl_overall_emotion

    try:
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        verified_faces_count = 0
        emotions = []

        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            try:
                analysis_results = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                verified_faces_count += 1
                emotions.append((analysis_results['dominant_emotion'], analysis_results['emotion'][analysis_results['dominant_emotion']]))

                result_text = f"Emotion: {analysis_results['dominant_emotion']}, Score: {analysis_results['emotion'][analysis_results['dominant_emotion']]:.2f}"
                result_label = tk.Label(result_frame, text=result_text, wraplength=250, justify="left", bg="#f0f0f0")
                result_label.pack(anchor='w', padx=10, pady=5)

            except Exception as deepface_error:
                print(f"DeepFace error: {deepface_error}")
        #error adjustment 

        lbl_faces_detected.config(text=f"Faces Detected: {verified_faces_count}")

        if emotions:
            overall_emotion, _ = max(set(emotions), key=emotions.count)
            overall_score = sum(score for emotion, score in emotions if emotion == overall_emotion) / len([score for emotion, score in emotions if emotion == overall_emotion])
            lbl_overall_emotion.config(text=f"Overall Emotion: {overall_emotion} (Avg. Score: {overall_score:.2f})")
        else:
            lbl_overall_emotion.config(text="Overall Emotion: N/A")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, width=250, height=250, bg="white", bd=2, relief="groove")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

open_button = tk.Button(main_frame, text="Open Image", command=open_file, bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

lbl_faces_detected = tk.Label(main_frame, text="Faces Detected: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_faces_detected.grid(row=2, column=0, sticky="w")

lbl_overall_emotion = tk.Label(main_frame, text="Overall Emotion: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_overall_emotion.grid(row=3, column=0, sticky="w")

result_frame = tk.Frame(main_frame, bg="#f0f0f0")
result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

root.mainloop()"""

#4th attempt-ABLE TO DETECT FACES CORRECT 
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np

def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        load_image(filename)
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame, lbl_faces_detected, lbl_overall_emotion

    try:
        image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for RetinaFace
        faces = detect_faces(image_rgb)

        verified_faces_count = 0
        emotion_scores = {}

        for (x, y, w, h) in faces:
            face_img = image_rgb[y:y+h, x:x+w]
            
            try:
                analysis_results = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                verified_faces_count += 1
                dominant_emotion = analysis_results['dominant_emotion']
                emotion_score = analysis_results['emotion'][dominant_emotion]

                if dominant_emotion in emotion_scores:
                    emotion_scores[dominant_emotion].append(emotion_score)
                else:
                    emotion_scores[dominant_emotion] = [emotion_score]

                result_text = f"Emotion: {dominant_emotion}, Score: {emotion_score:.2f}"
                result_label = tk.Label(result_frame, text=result_text, wraplength=250, justify="left", bg="#f0f0f0")
                result_label.pack(anchor='w', padx=10, pady=5)

            except Exception as deepface_error:
                print(f"DeepFace error: {deepface_error}")

        lbl_faces_detected.config(text=f"Faces Detected: {verified_faces_count}")

        if emotion_scores:
            overall_emotion = max(emotion_scores, key=lambda k: len(emotion_scores[k]))
            average_score = sum(emotion_scores[overall_emotion]) / len(emotion_scores[overall_emotion])
            lbl_overall_emotion.config(text=f"Overall Emotion: {overall_emotion} (Avg. Score: {average_score:.2f})")
        else:
            lbl_overall_emotion.config(text="Overall Emotion: N/A")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def detect_faces(image):
    faces = RetinaFace.detect_faces(image)
    faces_list = []
    if faces is not None:
        for key in faces:
            face = faces[key]
            if "facial_area" in face:
                try:
                    x, y, x2, y2 = face["facial_area"]
                    width, height = x2 - x, y2 - y
                    faces_list.append((x, y, width, height))
                except ValueError as e:
                    print(f"Error unpacking facial_area for {key}: {e}")
            else:
                print(f"'facial_area' not found for {key}")
    return faces_list

root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, width=250, height=250, bg="white", bd=2, relief="groove")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

open_button = tk.Button(main_frame, text="Open Image", command=open_file, bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

lbl_faces_detected = tk.Label(main_frame, text="Faces Detected: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_faces_detected.grid(row=2, column=0, sticky="w")

lbl_overall_emotion = tk.Label(main_frame, text="Overall Emotion: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_overall_emotion.grid(row=3, column=0, sticky="w")

result_frame = tk.Frame(main_frame, bg="#f0f0f0")
result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

root.mainloop()
"""

#5TH ATTEMPT WORKS SO SEXY SO SO SEXYYYYY
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np

def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        load_image(filename)
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo  # Keep a reference so it's not garbage collected
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame, lbl_faces_detected, lbl_overall_emotion

    try:
        image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for RetinaFace
        faces = detect_faces(image_rgb)

        verified_faces_count = 0
        emotion_scores = {}

        for (x, y, w, h) in faces:
            face_img = image_rgb[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))  # Resize for DeepFace
            
            try:
                analysis_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

                if isinstance(analysis_result, list) and len(analysis_result) > 0:
                    analysis_result = analysis_result[0]  # Take the first result
                    verified_faces_count += 1
                    dominant_emotion = analysis_result['dominant_emotion']
                    emotion_score = analysis_result['emotion'][dominant_emotion]

                    if dominant_emotion in emotion_scores:
                        emotion_scores[dominant_emotion].append(emotion_score)
                    else:
                        emotion_scores[dominant_emotion] = [emotion_score]

            except Exception as deepface_error:
                print(f"DeepFace error: {deepface_error}")

        lbl_faces_detected.config(text=f"Faces Detected: {verified_faces_count}")

        if emotion_scores:
            overall_emotion, average_score = calculate_overall_emotion(emotion_scores)
            lbl_overall_emotion.config(text=f"Overall Emotion: {overall_emotion} (Avg. Score: {average_score:.2f})")
        else:
            lbl_overall_emotion.config(text="Overall Emotion: N/A")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def detect_faces(image):
    faces = RetinaFace.detect_faces(image)
    faces_list = []
    if faces is not None:
        for key, value in faces.items():
            facial_area = value['facial_area']
            x, y, x2, y2 = facial_area
            faces_list.append((x, y, x2 - x, y2 - y))
    return faces_list

def calculate_overall_emotion(emotion_scores):
    emotions = sum(([emotion] * len(scores) for emotion, scores in emotion_scores.items()), [])
    dominant_emotion = max(set(emotions), key=emotions.count)
    average_score = np.mean(emotion_scores[dominant_emotion])
    return dominant_emotion, average_score

root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, width=250, height=250, bg="white", bd=2, relief="groove")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

open_button = tk.Button(main_frame, text="Open Image", command=open_file, bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

lbl_faces_detected = tk.Label(main_frame, text="Faces Detected: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_faces_detected.grid(row=2, column=0, sticky="w")

lbl_overall_emotion = tk.Label(main_frame, text="Overall Emotion: N/A", bg="#4CAF50", fg="white", padx=10, pady=5)
lbl_overall_emotion.grid(row=3, column=0, sticky="w")

result_frame = tk.Frame(main_frame, bg="#f0f0f0")
result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

root.mainloop()
"""

#6th attempt-the whole code

import tkinter as tk #for GUI
from tkinter import filedialog, messagebox
import cv2 #for image analysis 
from PIL import Image, ImageTk
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np
import matplotlib.pyplot as plt #for plotting the valence arousal plane 

def open_file():
    filename = filedialog.askopenfilename(title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
    if filename:
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        load_image(filename)
        analyze_and_display(filename)

def load_image(filepath):
    image = Image.open(filepath)
    image.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(image)
    
    canvas.image = photo  # Keep a reference so it's not garbage collected
    canvas.create_image(0, 0, anchor='nw', image=photo)

def analyze_and_display(filename):
    global result_frame, lbl_faces_detected, lbl_overall_emotion, overall_emotion, overall_score

    try:
        image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for RetinaFace
        faces = detect_faces(image_rgb)

        verified_faces_count = 0
        emotion_scores = {}

        for (x, y, w, h) in faces:
            face_img = image_rgb[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))  # Resize for DeepFace
            
            try:
                analysis_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

                if isinstance(analysis_result, list) and len(analysis_result) > 0:
                    analysis_result = analysis_result[0]  # Take the first result
                    verified_faces_count += 1 #this is for face deetction
                    dominant_emotion = analysis_result['dominant_emotion']
                    emotion_score = analysis_result['emotion'][dominant_emotion]

                    if dominant_emotion in emotion_scores:
                        emotion_scores[dominant_emotion].append(emotion_score)
                    else:
                        emotion_scores[dominant_emotion] = [emotion_score]

            except Exception as deepface_error:
                print(f"DeepFace error: {deepface_error}")

        lbl_faces_detected.config(text=f"Faces Detected: {verified_faces_count}")

        if emotion_scores:
            dom_emotion, avg_score = calculate_overall_emotion(emotion_scores)
            overall_emotion.set(dom_emotion)
            overall_score.set(avg_score)
            lbl_overall_emotion.config(text=f"Overall Emotion: {dom_emotion} (Avg. Score: {avg_score:.2f})")
        else:
            lbl_overall_emotion.config(text="Overall Emotion: N/A")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def detect_faces(image):
    faces = RetinaFace.detect_faces(image)
    faces_list = []
    if faces is not None:
        for key, value in faces.items():
            facial_area = value['facial_area']
            x, y, x2, y2 = facial_area
            faces_list.append((x, y, x2 - x, y2 - y))
    return faces_list

def calculate_overall_emotion(emotion_scores):
    emotions = sum(([emotion] * len(scores) for emotion, scores in emotion_scores.items()), [])
    dominant_emotion = max(set(emotions), key=emotions.count)
    average_score = np.mean(emotion_scores[dominant_emotion])
    return dominant_emotion, average_score

def plot_valence_arousal():
    emotion = overall_emotion.get()
    score = float(overall_score.get())
    
    # Define emotion placement in the valence-arousal space
    placement = {
        'happy': (score, score),
        'angry': (-score, score),
        'disgust': (-score, score),
        'sad': (-score, -score),
        'fear': (-score, -score),
        'surprise': (score, score),
        'neutral': (0, 0)
    }

    plt.figure(figsize=(6, 6))
    for em, pos in placement.items():
        if em == emotion:
            plt.scatter(*pos, label=f"{emotion} (Score: {score:.2f})", color='red')
        else:
            plt.scatter(*pos, alpha=0.2)
    plt.title("Valence-Arousal Space")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    plt.show()

root = tk.Tk()
root.title("Image Analysis")
root.configure(bg="#f0f0f0")

main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, width=250, height=250, bg="white", bd=2, relief="groove")
canvas.grid(row=0, column=0, columnspan=2, pady=10)

open_button = tk.Button(main_frame, text="Open Image", command=open_file, bg="#4CAF50", fg="black", padx=10, pady=5)
open_button.grid(row=1, column=0, columnspan=2, pady=10)

lbl_faces_detected = tk.Label(main_frame, text="Faces Detected: N/A", bg="white", fg="black", padx=10, pady=5)
lbl_faces_detected.grid(row=2, column=0, sticky="w")

lbl_overall_emotion = tk.Label(main_frame, text="Overall Emotion: N/A", bg="white", fg="black", padx=10, pady=5)
lbl_overall_emotion.grid(row=3, column=0, sticky="w")

result_frame = tk.Frame(main_frame, bg="#f0f0f0")
result_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

overall_emotion = tk.StringVar()
overall_score = tk.StringVar()

plot_button = tk.Button(main_frame, text="Valence-Arousal Graph", command=plot_valence_arousal, bg="#2196F3", fg="black", padx=10, pady=5)
plot_button.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
