import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2

# Crear el modelo de red neuronal (como en el código anterior)

# Compilar el modelo (como en el código anterior)

# Cargar el modelo entrenado (si ya tienes uno entrenado)
# model = tf.keras.models.load_model('modelo_reconocimiento_facial.h5')

# Abre la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesa la imagen de la cámara para que coincida con el formato esperado por el modelo
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0)
    
    # Utiliza el modelo para predecir
    prediction = model.predict(img)
    
    # Obtén el índice de la clase predicha
    predicted_class = np.argmax(prediction)
    
    # Dibuja un cuadro y muestra el resultado en la imagen de la cámara
    if predicted_class == 0:
        label = 'No Rostro'
    else:
        label = 'Rostro'
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Reconocimiento Facial', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
