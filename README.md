# Sistema de Diagnóstico de Enfermedades en Papa con Deep Learning

![Potato Disease Classification](https://www.mdpi.com/agriculture/agriculture-14-00386/article_deploy/html/images/agriculture-14-00386-g001.png)

## Contexto

Este proyecto tiene como objetivo desarrollar un sistema inteligente basado en redes neuronales convolucionales (CNN) para identificar automáticamente enfermedades en hojas de papa, ayudando a agricultores y técnicos agrícolas en el diagnóstico temprano de patologías como:

- **Tizón Temprano** (Early Blight)
- **Tizón Tardío** (Late Blight)
- **Hojas saludables** (Healthy)

## ¿Por qué es importante?

La papa (Solanum tuberosum) es uno de los cultivos más importantes a nivel mundial, especialmente en países como Perú. Sin embargo:

- Los métodos tradicionales de diagnóstico son subjetivos y requieren expertise
- El diagnóstico tardío puede causar pérdidas significativas en los cultivos
- Muchas zonas agrícolas carecen de acceso a especialistas fitosanitarios

Este sistema proporciona:
✅ Diagnóstico rápido y preciso (mayor al 90% de exactitud)  
✅ Plataforma accesible desde dispositivos móviles  
✅ Recomendaciones específicas para cada enfermedad  
✅ Reducción en el uso innecesario de pesticidas  

## Tecnologías Utilizadas

### Modelos de Deep Learning
- **EfficientNetB0**: Modelo ligero con alta eficiencia computacional
- **ResNet50V2**: Arquitectura profunda con conexiones residuales
- **Xception**: Basado en convoluciones separables por profundidad

### Framework y Herramientas
- TensorFlow/Keras para el desarrollo de modelos
- Streamlit para la interfaz web interactiva
- OpenCV para procesamiento de imágenes
- Google Colab para entrenamiento con GPU

### Métricas de Evaluación
- Exactitud (Accuracy)
- Precisión (Precision)
- Sensibilidad (Recall)
- F1-Score
- Coeficiente de Matthews (MCC)
- Coefciente de McNemar

## Dataset

El modelo fue entrenado con el dataset:  
[**Potato Disease Dataset**](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) disponible en Kaggle

**Características del dataset**:
- Total de imágenes: 2,152
- Distribución:
  - Early Blight: 1,000 imágenes
  - Late Blight: 1,000 imágenes
  - Healthy: 152 imágenes
- Formato: JPG
- Resolución: Variable (se redimensionaron a 224x224 píxeles)

## Metodología

1. **Preprocesamiento**:
   - Redimensionamiento a 224x224 píxeles
   - Normalización de valores de píxeles
   - Aumento de datos (rotaciones, cambios de brillo, etc.)

2. **Arquitectura del Modelo**:
   - Transfer Learning con modelos preentrenados en ImageNet
   - Capas personalizadas para clasificación
   - Fine-tuning de parámetros

3. **Entrenamiento**:
   - 30 épocas con early stopping
   - Optimizador Adam (learning rate = 0.0001)
   - Función de pérdida: categorical crossentropy

## Resultados

| Modelo        | Exactitud | Precisión | Recall  | F1-Score | MCC    |
|---------------|-----------|-----------|---------|----------|--------|
| EfficientNetB0| 78.63%    | 72.08%    | 78.63%  | 75.05%   | 61.82% |
| ResNet50V2    | 97.44%    | 97.45%    | 97.44%  | 97.40%   | 95.55% |
| Xception      | 94.59%    | 94.72%    | 94.59%  | 94.49%   | 90.59% |

## Interfaz Web

La aplicación permite:
- Visualizar los modelos de entrenamiento y dataset entrenado
- Subir imágenes de hojas de papa
- Obtener diagnóstico instantáneo
- Visualizar mapas de calor del diagnóstico de enfermedad
- Descargar reportes completos en PDF con recomendaciones y gráficos

![Interfaz de la Aplicación](/assets/interfaz.jpg) 

## Autores
1. José Andrés Farro Lagos - Universidad Nacional de Trujillo
2. Stefany Marisel Pérez Bazán - Universidad Nacional de Trujillo
**Asesor:** Dr. Juan Pedro Santos Fernández  
