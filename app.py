from flask import Flask, request, jsonify
from transformers import pipeline
from pyngrok import ngrok
import os

# Inicializamos el modelo de Hugging Face para clasificación de emociones
emotion_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Lista de emociones predefinidas
emotions = ['Felicidad', 'Tristeza', 'Enojo', 'Miedo', 'Sorpresa', 'Desconfianza', 'Neutralidad']

# Función para detectar la emoción
def detectar_emocion(texto):
    result = emotion_model(texto, candidate_labels=emotions)
    return result['labels'][0], result['scores'][0]

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal para analizar la emoción del usuario
@app.route('/analizar', methods=['POST'])
def analizar_emocion():
    data = request.json
    texto_usuario = data.get('mensaje', '')
    
    if not texto_usuario:
        return jsonify({'error': 'No se proporcionó mensaje'}), 400
    
    emocion, confianza = detectar_emocion(texto_usuario)
    
    # Respuesta con la emoción detectada y la confianza
    return jsonify({
        'emoción_detectada': emocion,
        'confianza': confianza
    })

if __name__ == '__main__':
    # Configuramos el puerto
    app.run(port=5000)
