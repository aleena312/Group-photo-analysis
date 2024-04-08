import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from deepface import DeepFace
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import matplotlib.pyplot as plt

def open_file():
    filename = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
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
    global lbl_faces_detected, lbl_overall_emotion, overall_emotion, overall_score

    try:
        image = cv2.imread(filename)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        context_info = analyze_context(image_rgb)
        analysis_results = DeepFace.analyze(img_path=filename, actions=['emotion'], enforce_detection=True, detector_backend='retinaface')
        emotion_scores = {}

        if isinstance(analysis_results, list):
            faces_detected = len(analysis_results)
            for result in analysis_results:
                process_emotion_result(result, emotion_scores)
        else:
            faces_detected = 1
            process_emotion_result(analysis_results, emotion_scores)

        lbl_faces_detected.config(text=f"Faces Detected: {faces_detected}")

        if emotion_scores:
            dom_emotion, avg_score = calculate_overall_emotion(emotion_scores, context_info)
            overall_emotion.set(dom_emotion)
            overall_score.set(f"{avg_score:.2f}")
            lbl_overall_emotion.config(text=f"Overall Emotion: {dom_emotion} (Avg. Score: {avg_score:.2f})")
        else:
            lbl_overall_emotion.config(text="Overall Emotion: N/A")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def process_emotion_result(analysis_result, emotion_scores):
    dominant_emotion = analysis_result['dominant_emotion']
    emotion_score = analysis_result['emotion'][dominant_emotion]
    if dominant_emotion in emotion_scores:
        emotion_scores[dominant_emotion].append(emotion_score)
    else:
        emotion_scores[dominant_emotion] = [emotion_score]

def analyze_context(image_rgb):
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_preprocessed = preprocess_input(np.expand_dims(image_resized, axis=0))
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    features = vgg16_model.predict(image_preprocessed)

    feature_mean = np.mean(features)
    print("mean feature:", feature_mean)

    context_info = {'Happy': 0, 'Sad': 0, 'Fear': 0, 'Surprise': 0, 'Angry': 0, 'Disgust': 0, 'Neutral': 0}
    if 2.0 <= feature_mean <= 2.9 or 1.81 <= feature_mean <= 2.8:
        context_info['Happy'] = 1
        context_info['Surprise'] = 1
    elif 1.3 <= feature_mean <= 1.7 or 1.7 <= feature_mean <= 1.8 or 1.8 <= feature_mean <= 2.1:
        context_info['Sad'] = 1
        context_info['Fear'] = 1
        context_info['Angry'] = 1
        context_info['Disgust'] = 1
    else:
        context_info['Neutral'] = 1

    print(context_info)
    return context_info

def calculate_overall_emotion(emotion_scores, context_info):
    avg_scores = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
    context_weight = 0.4  # Your defined context weight

    weighted_scores = {}
    for emotion, scores in emotion_scores.items():
        context_influence = context_info.get(emotion, 0) * context_weight *100
        weighted_scores[emotion] = len(scores) * avg_scores[emotion] + context_influence

    overall_dominant_emotion = max(weighted_scores, key=weighted_scores.get)
    overall_avg_score = weighted_scores[overall_dominant_emotion]

    return overall_dominant_emotion, overall_avg_score

def plot_valence_arousal():
    emotion = overall_emotion.get()
    score = float(overall_score.get())
    placement = {
        'happy': (score, score),  # High Arousal and High Valence (HAHV)
        'surprise': (score, score),  # High Arousal and High Valence (HAHV)
        'angry': (-score, score),  # High Arousal and Low Valence (HALV)
        'fear': (-score, score),  # High Arousal and Low Valence (HALV)
        'disgust': (-score, score),  # High Arousal and Low Valence (HALV)
        'sad': (-score, -score),  # Low Arousal and Low Valence (LALV)
        'neutral': (score, -score)  # Low Arousal and High Valence (LAHV)
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

    # Adding quadrant labels
    plt.text(0.05, 0.95, 'HALV', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, 'LAHV', horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.text(0.95, 0.95, 'HAHV', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.text(0.95, 0.05, 'LALV', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)

    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    plt.show()


root = tk.Tk()
root.title("Image Emotion Analyzer")
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
