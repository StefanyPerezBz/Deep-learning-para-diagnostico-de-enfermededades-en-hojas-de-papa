# 🌱 Sistema de Diagnóstico de Enfermedades en Hojas de Papa con Deep Learning

![Potato Disease Classification](./assets/hojas.png)

## 📌 Contexto

Este proyecto busca desarrollar un sistema inteligente basado en redes neuronales convolucionales (CNN) para identificar automáticamente enfermedades en hojas de papa, apoyando a agricultores y técnicos en el diagnóstico temprano de patologías como:

- **Hoja sana**
- **Tizón tardío** (Late blight)
- **Tizón temprano** (Early blight)

## 🚜 ¿Por qué es importante?

La papa (Solanum tuberosum) es uno de los cultivos más relevantes a nivel mundial, especialmente en países como Perú. Sin embargo:

- El diagnóstico tradicional es subjetivo y requiere experiencia.
- La detección tardía genera pérdidas significativas en los cultivos.
- Muchas zonas agrícolas carecen de especialistas fitosanitarios.

Con este sistema se logra:
- ✅ Diagnóstico rápido y preciso (**>90% de exactitud**)
- ✅ Acceso desde dispositivos móviles (vía interfaz web)
- ✅ Recomendaciones específicas por enfermedad
- ✅ Reducción del uso innecesario de pesticidas

## 🖥️ Interfaz
Una aplicación web interactiva construida con Streamlit, que incluye:
- Barra lateral izquierda con controles de configuración
- Área principal para visualizar resultados, gráficos y reportes
- Soporte multidioma: Español e Inglés

Funcionalidades principales:

- Selección del modelo de deep learning (EfficientNetB0, ResNet50V2, Xception, MobileNetV2, DenseNet121)
- Configuración de parámetros de entrenamiento (épocas, batch size, learning rate)
- Visualización del dataset con ejemplos e histogramas de distribución
- Entrenamiento con transfer learning
- Evaluación comparativa de modelos
- Análisis estadístico (ANOVA, McNemar, Tukey)
- Diagnóstico de imágenes subidas por el usuario
- Generación de reportes PDF con métricas y gráfico

## 📊 Resultados visualizados

- **Información del sistema** (hardware/software)
- **Estadísticas del dataset** (tamaño, balance de clases)
- **Métricas**: Accuracy, Precision, Recall, F1-score, **MCC**
- **Gráficos**:
  - Matrices de confusión
  - Curvas ROC
  - Curvas de aprendizaje
  - Comparación de modelos
- **Diagnóstico en tiempo real** para imágenes subidas

## 🛠️ Tecnologías Utilizadas

### Modelos de Deep Learning

- **CNN Personalizada**
- **ResNet50V2**
- **Xception**
- **MobileNetV2**
- **DenseNet121**

### Frameworks y librerías

- Python 3
- TensorFlow / Keras – entrenamiento de CNNs
- Streamlit – interfaz web interactiva
- OpenCV – procesamiento de imágenes
- Scikit-learn – métricas y análisis estadístico
- Matplotlib / Seaborn – visualización de resultados
- Google Colab - entrenamiento con GPU
- Ngrok – despliegue rápido en nube

### Métricas de Evaluación

- Exactitud (Accuracy)
- Precisión (Precision)
- Sensibilidad (Recall)
- F1-Score
- Coeficiente de Matthews (MCC)
- Coefciente de McNemar

## 📂 Dataset

El modelo fue entrenado con el dataset público: [**Potato Disease Dataset**](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) 

**Características**:

- Total: 2,152 imágenes
- Distribución:
  - Early Blight → 1,000
  - Late Blight → 1,000
  - Healthy → 152
- Formato: JPG
- Resolución: Variable (redimensionado a 224×224 px)

## 🗂️ Estructura del dataset
```
PlantVillage/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

## 🔬 Metodología

**Preprocesamiento**:
   - Redimensionamiento (224×224 px)
   - Normalización
   - Aumento de datos (data augmentation)

**Arquitectura del Modelo**:
- Transfer Learning con pesos de ImageNet
- Capas personalizadas para clasificación
- Fine-tuning parcial

**Funcionalidades principales**:
   - 30 épocas con early stopping
   - Optimizador: Adam (lr=0.0001)
   - Pérdida: categorical crossentropy
  
## 🎯 Interfaz
![Interfaz de la Aplicación](./assets/inicio.jpg)

## ⚠️ Requisitos para ejecutar

- Python 3
- Dependencias: tensorflow, streamlit, opencv-python, scikit-learn, matplotlib, seaborn, reportlab, ngrok
- Cuenta en Kaggle y Google Colab (opcional para entrenamiento con GPU)

## 📜 Licencia
MIT License – Ver LICENSE para detalles completos.

Nota: Proyecto desarrollado con fines academicos y de investigación.

## 👩‍💻 Autores

1. José Andrés Farro Lagos - Universidad Nacional de Trujillo
2. Stefany Marisel Pérez Bazán - Universidad Nacional de Trujillo
3.   **Asesor:** Dr. Juan Pedro Santos Fernández - Universidad Nacional de Trujillo

