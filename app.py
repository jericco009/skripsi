from flask import Flask, render_template, url_for, redirect, Response, jsonify
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Emotion counter and detection timestamp dictionary
emotion_count = { "Marah": 0, "Jijik": 0, "Takut": 0, "Senang": 0, "Netral": 0, "Sedih": 0, "Terkejut": 0 }
emotion_last_detected = { "Marah": 0, "Jijik": 0, "Takut": 0, "Senang": 0, "Netral": 0, "Sedih": 0, "Terkejut": 0 }
last_emotion = None
plot_path = 'static/plot.png'  # Location to save plot image
emotion_weights = {
    "Marah": -0.25, 
    "Jijik": 0.0, 
    "Takut": 0.0, 
    "Senang": 1.0, 
    "Netral": 0.5, 
    "Sedih": 0.25, 
    "Terkejut": 0.75
}

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Memuat bobot model yang sudah dilatih
model.load_weights('model.h5')

# Define plot_model_history function
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy plot
    axs[0].plot(model_history.history['accuracy'], label='train')
    axs[0].plot(model_history.history['val_accuracy'], label='val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='best')
    # Loss plot
    axs[1].plot(model_history.history['loss'], label='train')
    axs[1].plot(model_history.history['val_loss'], label='val')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='best')
    fig.savefig(plot_path)  # Save the plot as an image
    plt.close(fig)  # Close plot to free memory

@app.route('/train')
def train_model_route():
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    # Setup data generators
    train_dir = 'data/train'
    val_dir = 'data/test'
    batch_size = 64
    num_epoch = 10
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')

    model_info = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=num_epoch, validation_data=validation_generator, validation_steps=len(validation_generator))

    plot_model_history(model_info)

    # Collect epoch data
    epochs_data = [{"epoch": i + 1, "train_loss": model_info.history['loss'][i], "train_accuracy": model_info.history['accuracy'][i], "val_loss": model_info.history['val_loss'][i], "val_accuracy": model_info.history['val_accuracy'][i]} for i in range(num_epoch)]
    model.save_weights('model.h5')
    return render_template('training_dataset.html', epochs_data=epochs_data, plot_url=url_for('static', filename='plot.png'))

def detect_emotion(frame):
    global emotion_count, emotion_last_detected, last_emotion
    emotion_dict = {0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang", 4: "Netral", 5: "Sedih", 6: "Terkejut"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion_label = emotion_dict[maxindex]
     
        if emotion_label != last_emotion:
            last_emotion = emotion_label
            emotion_last_detected[emotion_label] = time.time()

        # Cek apakah ekspresi telah bertahan selama 3 detik
        if time.time() - emotion_last_detected[emotion_label] >= 3:
            # Jika sudah lebih dari 3 detik, tambahkan count ekspresi tersebut
            emotion_count[emotion_label] += 1
            emotion_last_detected[emotion_label] = time.time() 

        cv2.putText(frame, emotion_label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/monitoring')
def monitoring():
    global emotion_count
    # Reset emotion_count setiap kali route /monitoring diakses
    emotion_count = {
        "Marah": 0,
        "Jijik": 0,
        "Takut": 0,
        "Senang": 0,
        "Netral": 0,
        "Sedih": 0,
        "Terkejut": 0
    }
    return render_template("monitoring.html")

@app.route('/get_emotion_count')
def get_emotion_count():
    print("emotion_count:", emotion_count)  # Menampilkan nilai emotion_count
    return jsonify({
        'emotion_count': {
            0: emotion_count["Marah"], 
            1: emotion_count["Jijik"], 
            2: emotion_count["Takut"], 
            3: emotion_count["Senang"], 
            4: emotion_count["Netral"], 
            5: emotion_count["Sedih"], 
            6: emotion_count["Terkejut"]
        }
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kesimpulan_reaksi')
def kesimpulan_reaksi():
    global emotion_count

    # Ekspresi yang ingin ditampilkan saja
    selected_emotions = ["Marah", "Senang", "Netral", "Sedih", "Terkejut"]

    # Filter emotion_count agar hanya memuat ekspresi terpilih
    filtered_emotion_count = {emotion: emotion_count.get(emotion, 0) for emotion in selected_emotions}

    total_emotions = sum(filtered_emotion_count.values()) or 1  # Hindari pembagian dengan nol

    # Hitung skor kepuasan hanya dari emosi terpilih
    satisfaction_score = sum(
        filtered_emotion_count[emotion] * emotion_weights[emotion]
        for emotion in filtered_emotion_count
    ) / total_emotions

    satisfaction_percentage = satisfaction_score * 100

    if satisfaction_percentage > 80:
        satisfaction_level = "Sangat Puas"
    elif satisfaction_percentage > 60:
        satisfaction_level = "Puas"
    elif satisfaction_percentage > 40:
        satisfaction_level = "Cukup Puas"
    elif satisfaction_percentage > 20:
        satisfaction_level = "Kurang Puas"
    else:
        satisfaction_level = "Tidak Puas"

    emotions = list(filtered_emotion_count.keys())
    counts = list(filtered_emotion_count.values())
    colors = ['red', 'green', 'gray', 'blue', 'orange']

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        counts,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.4),
        pctdistance=0.85
    )

    ax.legend(wedges, emotions, title="Emosi", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('Persentase Ekspresi Emosi yang Terdeteksi')

    pie_chart_path = 'static/emotion_pie_chart.png'
    fig.savefig(pie_chart_path, bbox_inches='tight')
    plt.close(fig)

    return render_template(
        'kesimpulan_reaksi.html',
        emotion_count=filtered_emotion_count,
        total_emotions=sum(filtered_emotion_count.values()),
        satisfaction_percentage=satisfaction_percentage,
        satisfaction_level=satisfaction_level,
        plot_url=url_for('static', filename='emotion_pie_chart.png')
    )

# @app.route('/kesimpulan_reaksi')
# def kesimpulan_reaksi():
#     global emotion_count
#     # Menghitung total ekspresi yang terdeteksi
#     total_emotions = sum(emotion_count.values())
    
#     # Menghitung kepuasan pelanggan berdasarkan rumus bobot
#     satisfaction_score = sum(emotion_count[emotion] * emotion_weights[emotion] for emotion in emotion_count) / total_emotions

#     # Menghitung kepuasan dalam bentuk persentase
#     satisfaction_percentage = satisfaction_score * 100

#     # Menentukan tingkat kepuasan
#     if satisfaction_percentage > 80:
#         satisfaction_level = "Sangat Puas"
#     elif satisfaction_percentage > 60:
#         satisfaction_level = "Puas"
#     elif satisfaction_percentage > 40:
#         satisfaction_level = "Cukup Puas"
#     elif satisfaction_percentage > 20:
#         satisfaction_level = "Kurang Puas"
#     else:
#         satisfaction_level = "Tidak Puas"

#     # Membuat pie chart menggunakan matplotlib
#     emotions = list(emotion_count.keys())
#     counts = list(emotion_count.values())

#     fig, ax = plt.subplots()
#     ax.pie(counts, labels=emotions, autopct='%1.1f%%', colors=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink'])
#     ax.set_title('Persentase Ekspresi Emosi yang Terdeteksi')

#     # Simpan grafik sebagai gambar
#     pie_chart_path = 'static/emotion_pie_chart.png'
#     fig.savefig(pie_chart_path)
#     plt.close(fig)

#     return render_template('kesimpulan_reaksi.html', emotion_count=emotion_count, total_emotions=total_emotions,satisfaction_percentage=satisfaction_percentage, satisfaction_level=satisfaction_level, plot_url=url_for('static', filename='emotion_pie_chart.png'))


if __name__ == "__main__":
    app.run(debug=True)
