# Sistema de Diagnóstico de Enfermedades en Hojas de Papa con Deep Learning

![Potato Disease Classification](./assets/hojas.png)

## Contexto

Este proyecto tiene como objetivo desarrollar un sistema inteligente basado en redes neuronales convolucionales (CNN) para identificar automáticamente enfermedades en hojas de papa, ayudando a agricultores y técnicos agrícolas en el diagnóstico temprano de patologías como:

- **Hoja sana**
- **Tizón tardío** (Late blight)
- **Tizón temprano** (Early blight)

## ¿Por qué es importante?

La papa (Solanum tuberosum) es uno de los cultivos más importantes a nivel mundial, especialmente en países como Perú. Sin embargo:

- Los métodos tradicionales de diagnóstico son subjetivos y requieren expertise
- El diagnóstico tardío puede causar pérdidas significativas en los cultivos
- Muchas zonas agrícolas carecen de acceso a especialistas fitosanitarios

Este sistema proporciona:

- Diagnóstico rápido y preciso (mayor al 90% de exactitud)
- Plataforma accesible desde dispositivos móviles
- Recomendaciones específicas para cada enfermedad
- Reducción en el uso innecesario de pesticidas

## Tecnologías Utilizadas

### Modelos de Deep Learning

- **EfficientNetB0**: Modelo ligero con alta eficiencia computacional
- **ResNet50V2**: Arquitectura profunda con conexiones residuales
- **Xception**: Basado en convoluciones separables por profundidad
- **MobileNetV2**: Optimizado para móviles
- **DenseNet121**

### Requisitos Previos
- Cuenta en Google Colab
- Cuenta en Kaggle (para descargar el dataset)
- Google Drive (para almacenamiento)

### Framework y Herramientas

- Python 3
- TensorFlow/Keras
- Streamlit (Interfaz web)
- OpenCV (Procesamiento de imágenes)
- Scikit-learn (Métricas y evaluación)
- Matplotlib/Seaborn (Visualizaciones)
- Google Colab (Entrenamiento con GPU)
- Ngrok (Túneles para demostración)

### Métricas de Evaluación

- Exactitud (Accuracy)
- Precisión (Precision)
- Sensibilidad (Recall)
- F1-Score
- Coeficiente de Matthews (MCC)
- Coefciente de McNemar

## Dataset

El modelo fue entrenado con el dataset público:  
[**Potato Disease Dataset**](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) disponible en la plataforma Kaggle

**Características del dataset**:

- Total de imágenes: 2,152
- Distribución:
  - Early Blight: 1,000 imágenes
  - Late Blight: 1,000 imágenes
  - Healthy: 152 imágenes
- Formato: JPG
- Resolución: Variable (se redimensionaron a 224x224 píxeles)

## Metodología

**Preprocesamiento**:
   - Redimensionamiento a 224x224 píxeles
   - Normalización de valores de píxeles
   - Aumento de datos (rotaciones, cambios de brillo, etc.)

**Arquitectura del Modelo**:
- Transfer Learning con modelos preentrenados en ImageNet
- Capas personalizadas para clasificación
- Fine-tuning de parámetros

**Funcionalidades principales**:
   - 30 épocas con early stopping
   - Optimizador Adam (learning rate = 0.0001)
   - Función de pérdida: categorical crossentropy
  
## Diagnóstico
- Subida de imágenes para predicción
- Niveles de confianza por clase
- Generación de reportes en PDF

## Reportes
- Técnico (especificaciones del sistema)
- Entrenamiento (métricas detalladas)
- Visual (gráficos interactivos)
- Diagnóstico (por imagen analizada)

## Soporte Multidioma
Disponible en:
- Español (es)
- Inglés (en)

## Interfaz
![Interfaz de la Aplicación](./assets/interfaz.jpg)

## Resultados

| Modelo         | Exactitud | Precisión | Recall | F1-Score | MCC    |
| -------------- | --------- | --------- | ------ | -------- | ------ |
| EfficientNetB0 | 92.29%    | 98.29%    | 98.29% | 98.28%   | 97.03% |
| ResNet50V2     | 98.58%    | 98.57%    | 98.58% | 98.57%   | 97.53% |
| Xception       | 96.01%    | 96.15%    | 96.01% | 95.96%   | 93.08% |


## Autores

1. José Andrés Farro Lagos - Universidad Nacional de Trujillo
2. Stefany Marisel Pérez Bazán - Universidad Nacional de Trujillo
3.   **Asesor:** Dr. Juan Pedro Santos Fernández
