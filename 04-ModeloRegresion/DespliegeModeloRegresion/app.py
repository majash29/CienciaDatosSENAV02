from flask import Flask, render_template, request
import joblib

# Cargar el modelo entrenado
model = joblib.load('models/modelo_regresion.pkl')  # Cargar el modelo previamente guardado

# Crear una aplicación Flask
app = Flask(__name__)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    colegio = float(request.form['colegio'])
    estrato = float(request.form['estrato'])
    nmat = float(request.form['nmat'])
    nlecto = float(request.form['nlecto'])
    apoyo = float(request.form['apoyo'])
    grado = float(request.form['grado'])
    
    # Realizar una predicción de probabilidades utilizando el modelo cargado
    pred_probabilities = model.predict_proba([[colegio, estrato, nmat, nlecto, apoyo, grado]])
    
    # Obtener los nombres de las clases (Deserción, Alerta, Buen estudiante)
    class_names = model.classes_

    # Crear un mensaje con las probabilidades de cada clase para la predicción futura
    mensaje = ""
    for i, class_name in enumerate(class_names):
        prob = pred_probabilities[0, i] * 100
        mensaje += f"Probabilidad de {class_name}: {round(prob, 2)}% <br/>"

    # Renderizar la plantilla 'result.html' y pasar el mensaje a la plantilla
    return render_template('result.html', pred=mensaje)

# Iniciar la aplicación si este script es el punto de entrada
if __name__ == '__main__':
    app.run()  # Iniciar la aplicación Flask
