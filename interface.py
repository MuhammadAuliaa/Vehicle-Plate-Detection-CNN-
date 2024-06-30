import cv2 as cv
import numpy as np
import streamlit as st
import cv2 as cv
import easyocr
import matplotlib.pyplot as plt
import io
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Function to train the CNN model
def train_cnn_model(data_dir, img_height, img_width, batch_size, epochs, model_name):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    
    num_classes = 72

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the trained model with the provided name
    model.save(f'{model_name}.h5')

    return history


def main():
    st.title("DETEKSI PLAT KENDARAAN (CNN) 🛵")
    over_theme = {'txc_inactive': '#FFFFFF'}

    navigation = option_menu(None, ["⛺︎ DASHBOARD", "⚙️ TRAINING KARAKTER (CNN)", "🕵🏻‍♂️ DETEKSI" ],
                             icons=[":flag-ea:", "none", 'none', 'none'],
                             menu_icon="cast", default_index=0, orientation="horizontal")
    
    if navigation == "⛺︎ DASHBOARD":
        st.header("Deskripsi Program")
        st.write("Selamat datang di Dashboard Deteksi Plat Kendaraan! Program ini menggunakan Convolutional Neural Network (CNN) untuk mendeteksi plat kendaraan pada gambar yang diunggah. Metode deteksi menggunakan teknik edge detection (kontur) untuk mengidentifikasi kontur plat secara presisi.")

        image_path_1 = "interfaceImg/gambar1.png" 

        col1, col2 = st.columns(2)  # Mengganti st.beta_columns dengan st.columns

        with col1:
            st.image(image_path_1, use_column_width=True)

        with col2:
            st.write("Setelah Anda mengunggah gambar kendaraan, klik tombol DETEKSI untuk memulai proses analisis. Dashboard ini dirancang untuk memberikan hasil deteksi yang cepat dan akurat, memanfaatkan kecanggihan algoritma CNN untuk memberikan pengalaman deteksi plat kendaraan yang optimal. Selamat mencoba!")

        st.write("SISTEM ini menawarkan kemudahan dalam mengunggah gambar kendaraan yang ingin Anda analisis. Setelah proses deteksi selesai, hasilnya akan ditampilkan dalam bentuk visualisasi yang mencakup kontur plat kendaraan yang terdeteksi. Dengan menggunakan algoritma CNN, program ini dapat memahami struktur dan pola unik pada gambar, memungkinkan identifikasi yang akurat bahkan pada kondisi pencahayaan yang berbeda. Selain itu, Anda dapat melihat hasil deteksi dalam bentuk gambar yang telah diubah ukurannya untuk memudahkan analisis. Selamat menjelajahi fitur-fitur deteksi canggih ini!")

    elif navigation == "⚙️ TRAINING KARAKTER (CNN)":
        # Create two columns for displaying inputs side by side
        col1, col2 = st.columns(2)

        # Input parameters manually
        data_dir = col1.text_input("Data directory", "dataset/")
        img_height = col2.number_input("Image height", value=40)

        # Input parameters manually
        img_width = col1.number_input("Image width", value=40)
        batch_size = col2.number_input("Batch size", value=32)

        # Input parameters manually
        epochs = st.number_input("Epochs", value=10)
        
        # Input for the model name
        model_name = st.text_input("Model name", "my_model2")

        # Button to trigger training
        if st.button("Train Model"):
            # Display a spinner during training
            with st.spinner("Training in progress..."):
                # Train the CNN model
                history = train_cnn_model(data_dir, img_height, img_width, batch_size, epochs, model_name)

            # Display final accuracy after training
            final_accuracy = history.history['accuracy'][-1]
            st.success(f"Training completed! Final accuracy: {final_accuracy:.4f}")

            # Plot training history if needed
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(epochs)

    
    elif navigation == "🕵🏻‍♂️ DETEKSI":
        # Tambahkan kode deteksi plat di sini
        plat_path = st.file_uploader("Upload Gambar Plat Kendaraan", type=["jpg", "jpeg", "png"])

        if plat_path:
            # Convert the uploaded image file to NumPy array
            img = np.asarray(bytearray(plat_path.read()), dtype=np.uint8)
            img = cv.imdecode(img, 1)  # 1 indicates loading the image in color (BGR)

            # Check if the image is loaded successfully
            if img is None:
                st.error("Error: Image not loaded. Please check the file path.")
            else:
                # Display the uploaded image
                # st.image(img, caption="Uploaded Image", use_column_width=True)
                st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True) 

                # Button to start detection
                if st.button("Deteksi", key="detect_button", help="Klik untuk memulai deteksi"):
                    # Resize the image
                    img = cv.resize(img, (int(img.shape[1] * 0.4), int(img.shape[0] * 0.4)))

                    # Convert from BGR to grayscale
                    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                    # Lighting normalization
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))

                    # Opening operation on grayscale image
                    img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)

                    # Lighting normalization
                    img_norm = img_gray - img_opening

                    # Adaptive thresholding for both white and black plates
                    _, img_norm_bw = cv.threshold(img_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                    # Visualization for checking
                    (thresh, img_without_norm_bw) = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                    fig = plt.figure(figsize=(10, 7))
                    row_fig = 2
                    column_fig = 2

                    # ... (kode deteksi plat lainnya)

                    # Display the processed images using Streamlit
                    st.subheader("PREPROCESSING PLAT KENDARAAN")

                    # Display the original image
                    # st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

                    # Display other processed images
                    # st.image(img_gray, caption="Grayscale Image", use_column_width=True)
                    # st.image(img_without_norm_bw, caption="Without Normalization", use_column_width=True)
                    # st.image(img_norm_bw, caption="With Normalization", use_column_width=True)

                    # Create two columns for displaying images side by side
                    col1, col2 = st.columns(2)

                    # Display the original image in the first column
                    col1.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

                    # Display the processed image with normalization in the second column
                    col2.image(img_norm_bw, caption="With Normalization", use_column_width=True)

                    # ------------------
                    # Deteksi plat menggunakan contours

                    # dapatkan contours dari citra kendaraan
                    contours_vehicle, hierarchy = cv.findContours(img_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get the contour for every area

                    # index contour yang berisi kandidat plat nomor
                    index_plate_candidate = []

                    # index counter dari setiap contour di contours_vehichle
                    index_counter_contour_vehicle = 0

                    # filter setiap contour untuk mendapatkan kandidat plat nomor
                    for contour_vehicle in contours_vehicle:
                        
                        # dapatkan posisi x, y, nilai width, height, dari contour
                        x,y,w,h = cv.boundingRect(contour_vehicle)

                        # dapatkan nilai aspect rationya
                        aspect_ratio = w/h

                        # dapatkan kandidat plat nomornya apabila:
                        # 1. lebar piksel lebih dari atau sama dengan 200 piksel
                        # 2. aspect rationya kurang dari atau sama dengan 4
                        if w >= 200 and aspect_ratio <= 4 : 
                            
                            # dapatkan index kandidat plat nomornya
                            index_plate_candidate.append(index_counter_contour_vehicle)
                        
                        # increment index counter dari contour
                        index_counter_contour_vehicle += 1

                    # buat duplikat citra RGB dan BW kendaraan untuk menampilkan lokasi plat
                    img_show_plate = img.copy() 
                    img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)

                    if len(index_plate_candidate) == 0:
                        # tampilkan peringatan plat nomor tidak terdeteksi
                        print("Plat nomor tidak ditemukan")

                    # jika jumlah kandidat plat sama dengan 1
                    elif len(index_plate_candidate) == 1:

                        # dapatkan lokasi untuk pemotongan citra plat
                        x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
                        
                        # gambar kotak lokasi plat nomor di citra RGB
                        cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

                        # gambar kotak lokasi plat nomor di citra BW
                        cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

                        # crop citra plat 
                        img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
                    else:
                        print('Dapat dua lokasi plat, pilih lokasi plat kedua')

                        # dapatkan lokasi untuk pemotongan citra plat
                        x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])

                        # gambar kotak lokasi plat nomor di citra RGB
                        cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

                        # gambar kotak lokasi plat nomor di citra BW
                        cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

                        # crop citra plat 
                        img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]

                    st.subheader("Deteksi Karakter Plat Nomor")

                    # Display the images related to character detection
                    # st.image(cv.cvtColor(img_show_plate_bw, cv.COLOR_BGR2RGB), caption="Lokasi Plat Nomor BW", use_column_width=True)
                    st.image(cv.cvtColor(img_show_plate, cv.COLOR_BGR2RGB), caption="Lokasi Plat Nomor", use_column_width=True)
                    st.image(img_plate_gray, caption="Hasil Crop Plat Nomor", use_column_width=True)
                    
                    # ========== SEGMENTASI ==========
                    (thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

                    # hasil dari konversi BW tidak terlalu mulus, 
                    # ada bagian-bagian kecil yang tidak diinginkan yang mungkin bisa mengganggu
                    # maka hilangkan area yang tidak diinginkan dengan operasi opening

                    # buat kernel dengan bentuk cross dan ukuran 3x3
                    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

                    # cv.imshow("sebelum open", img_plate_bw)

                    # lakukan operasi opening dengan kernel di atas
                    img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel) # apply morph open

                    # cv.imshow("sesudah open", img_plate_bw)

                    # Segmentasi karakter menggunakan contours
                    # dapatkan kontur dari plat nomor
                    contours_plate, hierarchy = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

                    # index contour yang berisi kandidat karakter
                    index_chars_candidate = [] #index

                    # index counter dari setiap contour di contours_plate
                    index_counter_contour_plate = 0 #idx

                    # duplikat dan ubah citra plat dari gray dan bw ke rgb untuk menampilkan kotak karakter
                    img_plate_rgb = cv.cvtColor(img_plate_gray,cv.COLOR_GRAY2BGR)
                    img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)

                    # Mencari kandidat karakter
                    for contour_plate in contours_plate:

                        # dapatkan lokasi x, y, nilai width, height dari setiap kontur plat
                        x_char,y_char,w_char,h_char = cv.boundingRect(contour_plate)
                        
                        # Dapatkan kandidat karakter jika:
                        #   tinggi kontur dalam rentang 40 - 60 piksel
                        #   dan lebarnya lebih dari atau sama dengan 10 piksel 
                        if h_char >= 40 and h_char <= 60 and w_char >=10:

                            # dapatkan index kandidat karakternya
                            index_chars_candidate.append(index_counter_contour_plate)

                            # gambar kotak untuk menandai kandidat karakter
                            cv.rectangle(img_plate_rgb,(x_char,y_char),(x_char+w_char,y_char+h_char),(0,255,0),5)
                            cv.rectangle(img_plate_bw_rgb,(x_char,y_char),(x_char+w_char,y_char+h_char),(0,255,0),5)

                        index_counter_contour_plate += 1

                    # tampilkan kandidat karakter
                    # cv.imshow('Kandidat Karakter',img_plate_rgb)

                    if index_chars_candidate == []:

                        # tampilkan peringatan apabila tidak ada kandidat karakter
                        print('Karakter tidak tersegmentasi')
                    else:

                        # untuk menyimpan skor setiap karakter pada kandidat
                        score_chars_candidate = np.zeros(len(index_chars_candidate))

                        # untuk counter index karakter
                        counter_index_chars_candidate = 0

                        # bandingkan lokasi y setiap kandidat satu dengan kandidat lainnya
                        for chars_candidateA in index_chars_candidate:
                            
                            # dapatkan nilai y dari kandidat A
                            xA,yA,wA,hA = cv.boundingRect(contours_plate[chars_candidateA])
                            for chars_candidateB in index_chars_candidate:

                                # jika kandidat yang dibandikan sama maka lewati
                                if chars_candidateA == chars_candidateB:
                                    continue
                                else:
                                    # dapatkan nilai y dari kandidat B
                                    xB,yB,wB,hB = cv.boundingRect(contours_plate[chars_candidateB])

                                    # cari selisih nilai y kandidat A dan kandidat B
                                    y_difference = abs(yA - yB)

                                    # jika perbedaannya kurang dari 11 piksel
                                    if y_difference < 11:
                                        
                                        # tambahkan nilai score pada kandidat tersebut
                                        score_chars_candidate[counter_index_chars_candidate] = score_chars_candidate[counter_index_chars_candidate] + 1 

                            # lanjut ke kandidat lain
                            counter_index_chars_candidate += 1

                        print(score_chars_candidate)

                        # untuk menyimpan karakter
                        index_chars = []

                        # counter karakter
                        chars_counter = 0

                        # dapatkan karakter, yaitu yang memiliki score tertinggi
                        for score in score_chars_candidate:
                            if score == max(score_chars_candidate):

                                # simpan yang benar-benar karakter
                                index_chars.append(index_chars_candidate[chars_counter])
                            chars_counter += 1

                        # duplikat dan ubah ke rgb untuk menampilkan urutan karakter yang belum terurut
                        img_plate_rgb2 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)

                        # tampilkan urutan karakter yang belum terurut
                        for char in index_chars:
                            x, y, w, h = cv.boundingRect(contours_plate[char])
                            cv.rectangle(img_plate_rgb2,(x,y),(x+w,y+h),(0,255,0),5)
                            cv.putText(img_plate_rgb2, str(index_chars.index(char)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)
                        
                        # untuk menyimpan koordinat x setiap karakter
                        x_coors = []

                        for char in index_chars:
                            # dapatkan nilai x
                            x, y, w, h = cv.boundingRect(contours_plate[char])

                            # dapatkan nilai sumbu x
                            x_coors.append(x)

                        # urutkan sumbu x dari terkecil ke terbesar
                        x_coors = sorted(x_coors)

                        # untuk menyimpan karakter
                        index_chars_sorted = []

                        # urutkan karakternya berdasarkan koordinat x yang sudah diurutkan
                        for x_coor in x_coors:
                            for char in index_chars:

                                # dapatkan nilai koordinat x karakter
                                x, y, w, h = cv.boundingRect(contours_plate[char])

                                # jika koordinat x terurut sama dengan koordinat x pada karakter
                                if x_coors[x_coors.index(x_coor)] == x:

                                    # masukkan karakternya ke var baru agar mengurut dari kiri ke kanan
                                    index_chars_sorted.append(char)

                        # duplikat dan ubah ke rgb untuk menampilkan yang benar-benar karakter
                        img_plate_rgb3 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)

                        # Gambar kotak untuk menandai karakter yang terurut dan tambahkan teks urutannya
                        for char_sorted in index_chars_sorted:

                            # dapatkan nilai x, y, w, h dari karakter terurut
                            x,y,w,h = cv.boundingRect(contours_plate[char_sorted])

                            # gambar kotak yang menandai karakter terurut
                            cv.rectangle(img_plate_rgb3,(x,y),(x+w,y+h),(0,255,0),5)

                            # tambahkan teks urutan karakternya
                            cv.putText(img_plate_rgb3, str(index_chars_sorted.index(char_sorted)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)
                    # ========== AKHIR SEGMENTASI ==========

                    # tinggi dan lebar citra untuk test
                    img_height = 40 
                    img_width = 40

                    # klas karakter
                    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

                    # load model yang sudah terlatih
                    model = keras.models.load_model('my_model')

                    # untuk menyimpan string karakter
                    num_plate = []

                    for char_sorted in index_chars_sorted:
                        x,y,w,h = cv.boundingRect(contours_plate[char_sorted])

                        # potong citra karakter
                        char_crop = cv.cvtColor(img_plate_bw[y:y+h,x:x+w], cv.COLOR_GRAY2BGR)

                        # resize citra karakternya
                        char_crop = cv.resize(char_crop, (img_width, img_height))

                        # preprocessing citra ke numpy array
                        img_array = keras.preprocessing.image.img_to_array(char_crop)

                        # agar shape menjadi [1, h, w, channels]
                        img_array = tf.expand_dims(img_array, 0)

                        # buat prediksi
                        predictions = model.predict(img_array)
                        score = tf.nn.softmax(predictions[0]) 

                        num_plate.append(class_names[np.argmax(score)])
                        print(class_names[np.argmax(score)], end='')

                    # Gabungkan string pada list
                    plate_number = ''
                    for a in num_plate:
                        plate_number += a

                    st.subheader("Hasil Deteksi dan Pembacaan Plat")

                    # Display the final result with detected plate number
                    st.image(cv.cvtColor(img_show_plate, cv.COLOR_BGR2RGB), caption=f"Detected Plate Number: {plate_number}", use_column_width=True)



if __name__ == "__main__":
    main()