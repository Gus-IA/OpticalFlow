# 🏀 Optical Flow con RAFT y PyTorch

Este proyecto muestra cómo calcular **Optical Flow (flujo óptico)** entre pares de frames de un video utilizando el modelo **RAFT (Recurrent All-Pairs Field Transforms)** disponible en `torchvision`.

A partir de un video de ejemplo, se seleccionan frames consecutivos, se procesan con un modelo preentrenado y se visualiza el movimiento entre imágenes mediante mapas de color.

---

## 📌 ¿Qué se aprendió?

Con este código se aprendió a:

- Descargar y cargar un video usando `torchvision`
- Extraer y seleccionar frames específicos de un video
- Trabajar con tensores de imágenes en PyTorch
- Preprocesar imágenes para modelos de visión por computadora
- Utilizar modelos preentrenados de **Optical Flow**
- Ejecutar inferencia en GPU (CUDA) si está disponible
- Interpretar la salida de un modelo RAFT
- Convertir flujo óptico en imágenes visuales
- Visualizar resultados con `matplotlib`

---

## 🧠 Conceptos Importantes

### 🔹 Optical Flow
El **flujo óptico** describe el movimiento aparente de los píxeles entre dos imágenes consecutivas. Es una técnica clave en:
- Análisis de video
- Seguimiento de objetos
- Visión por computadora
- Robótica y conducción autónoma

### 🔹 Modelo RAFT
**RAFT** es un modelo de deep learning de última generación para estimación de flujo óptico, que se caracteriza por:
- Alta precisión
- Uso de múltiples iteraciones para refinar el resultado
- Comparaciones densas entre todos los píxeles de las imágenes

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
