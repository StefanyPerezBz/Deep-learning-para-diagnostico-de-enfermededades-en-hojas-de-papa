'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autores: Stefany Marisel Pérez Bazán, José Andrés Farro Lagos
Fecha: 14/07/2025
Institución: UNT | Universidad Nacional de Trujillo

Este archivo forma parte del Artículo de Investigación "Sistema de Diagnóstico de Enfermedades en Papa con Deep Learning".
Los alumnos a cargo de este proyecto son declarados como autores en las líneas anteriores.
El tutor del proyecto fue el Dr. Juan Pedro Santos Fernández.

🔍 Qué es este fichero:
Este fichero contiene la configuración inicial y las funciones principales para el sistema de diagnóstico de enfermedades en hojas de papa.
Incluye la carga de datos, creación de modelos, entrenamiento, evaluación y generación de reportes
con gráficos y análisis estadísticos.

Modelos preentrenados: EfficientNetB0, ResNet50V2 y Xception (transfer learning).
Herramientas: TensorFlow/Keras, scikit-learn, OpenCV, Pandas, Matplotlib/Seaborn.
Estadística avanzada: Pruebas ANOVA, Tukey, McNemar para comparar modelos.

Funcionalidades destacadas:
- Entrenamiento/evaluación de modelos.
- Visualización de métricas (matrices de confusión, curvas de aprendizaje, mapas de calor).
- Pruebas inferenciales para comparar modelos.
- Generación de reportes PDF con análisis detallado.
- Interfaz amigable con Streamlit.
- Soporte para aumento de datos y early stopping.
- Carga de modelos guardados para reuso.
- Configuración flexible a través de la barra lateral.
- Recomendaciones personalizadas basadas en diagnósticos.
- Generación de gráficos y análisis estadísticos en PDF.

Diseñado para ejecutarse en Google Colab (por las rutas como /content/drive/... y GPU T4).

#########################################################################################################################
'''

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from scipy import stats
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import time
import shutil
from reportlab.platypus.flowables import Image as PlatypusImage
from itertools import combinations

# =============================================
# CONFIGURACIÓN INICIAL
# =============================================

# Configurar diseño de la página
st.set_page_config(
    page_title="Diagnóstico de Enfermedades en Hojas de Papa",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con estilo
st.title("🥔 Diagnóstico de Enfermedades en Hojas de Papa")
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
.recommendation {
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 5px solid #4e79a7;
}
.recommendation-header {
    color: #2e5b8e;
    font-weight: bold;
    margin-bottom: 10px;
}
.recommendation-item {
    margin-bottom: 8px;
    padding-left: 10px;
    border-left: 3px solid #a7c5eb;
}
.model-card {
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.anova-result {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.tukey-table {
    margin-top: 15px;
}
.significant {
    background-color: #c8e6c9 !important;
}
.not-significant {
    background-color: #ffcdd2 !important;
}
.heatmap-container {
    margin-top: 20px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Esta aplicación utiliza inteligencia artificial para identificar enfermedades en hojas de papa con modelos avanzados de deep learning.</p>', unsafe_allow_html=True)

# =============================================
# CONFIGURACIÓN DEL MODELO (SIDEBAR)
# =============================================

with st.sidebar:
    st.header("⚙️ Configuración")

    # Configuración del dataset
    dataset_path = st.text_input("Ruta del dataset", "/content/drive/MyDrive/Colab_Data/potato_data/PlantVillage")

    # Parámetros del modelo
    st.subheader("Parámetros de Entrenamiento")
    test_size = st.slider("Tamaño de validación (%)", 10, 40, 20)
    epochs = st.slider("Épocas", 5, 100, 30)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=0.01, value=0.001, step=0.0001, format="%.5f")

    # Selección de modelos
    st.subheader("Modelos Disponibles")
    MODELS = {
        "EfficientNetB0": EfficientNetB0,
        "ResNet50V2": ResNet50V2,
        "Xception": Xception
    }
    selected_models = st.multiselect(
        "Selecciona modelos a evaluar",
        list(MODELS.keys()),
        default=["EfficientNetB0"],
        max_selections=3
    )

    # Opciones avanzadas
    with st.expander("Opciones Avanzadas"):
        use_augmentation = st.checkbox("Aumento de datos", True)
        use_early_stopping = st.checkbox("Early Stopping", True)
        load_saved_models = st.checkbox("Intentar cargar modelos guardados primero", True)

# =============================================
# FUNCIONES PRINCIPALES
# =============================================

def create_model(base_model_name, num_classes, learning_rate):
    """Crea un modelo de transfer learning"""
    base_model = MODELS[base_model_name](
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3))

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def load_data(dataset_path):
    """Carga y preprocesa los datos"""
    try:
        classes = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])

        images = []
        labels = []

        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(dataset_path, class_name)
            for img_file in os.listdir(class_path)[:800]:  # Limitar para demo
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    continue

        X = np.array(images, dtype='float32')
        y = np.array(labels)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y)

        # Normalizar
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # One-hot encoding
        y_train = to_categorical(y_train, len(classes))
        y_test = to_categorical(y_test, len(classes))

        return X_train, X_test, y_train, y_test, classes

    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None, None, None, None, None

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Entrena un modelo con los datos proporcionados"""
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001))

    # Data augmentation
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0)
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0)

    return model, history

def plot_confusion_matrix(y_true, y_pred, classes):
    """Genera y muestra una matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predicciones')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    return plt.gcf()

def plot_heatmap(pred_probs, classes, model_name):
    """Genera un mapa de calor de las probabilidades de predicción"""
    plt.figure(figsize=(10, 3))
    sns.heatmap([pred_probs], 
                annot=True, 
                fmt=".2%", 
                cmap="YlGnBu",
                xticklabels=classes, 
                yticklabels=[model_name],
                cbar=False)
    plt.title("Distribución de Probabilidades por Clase", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()

def plot_learning_curves(history):
    """Genera gráficas de curvas de aprendizaje"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfica de precisión
    ax1.plot(history.history['accuracy'], label='Entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión del Modelo')
    ax1.set_ylabel('Precisión')
    ax1.set_xlabel('Época')
    ax1.legend()

    # Gráfica de pérdida
    ax2.plot(history.history['loss'], label='Entrenamiento')
    ax2.plot(history.history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo')
    ax2.set_ylabel('Pérdida')
    ax2.set_xlabel('Época')
    ax2.legend()

    plt.tight_layout()
    return fig

def statistical_tests(y_test, preds_dict):
    """Realiza pruebas estadísticas comparando modelos"""
    results = {
        'pairwise': {},
        'global': {}
    }
    model_names = list(preds_dict.keys())

    # 1. Pruebas por pares (McNemar y t-test)
    if len(model_names) >= 2:
        for (name1, pred1), (name2, pred2) in combinations(preds_dict.items(), 2):
            # McNemar
            model1_correct = (pred1 == y_test)
            model2_correct = (pred2 == y_test)

            table = [[sum((model1_correct & model2_correct)), sum((model1_correct & ~model2_correct))],
                     [sum((~model1_correct & model2_correct)), sum((~model1_correct & ~model2_correct))]]

            mcnemar_result = mcnemar(table, exact=False)

            # t-test pareado para accuracy
            acc1 = accuracy_score(y_test, pred1)
            acc2 = accuracy_score(y_test, pred2)
            t_stat, p_val = ttest_rel([acc1]*len(y_test), [acc2]*len(y_test))

            results['pairwise'][f"{name1}_vs_{name2}"] = {
                'mcnemar': {
                    'statistic': mcnemar_result.statistic,
                    'p-value': mcnemar_result.pvalue,
                    'significativo': mcnemar_result.pvalue < 0.05
                },
                't-test': {
                    'statistic': t_stat,
                    'p-value': p_val,
                    'significativo': p_val < 0.05
                }
            }

    # 2. ANOVA y Tukey (para 3+ modelos)
    if len(model_names) >= 3:
        # Preparamos datos para ANOVA (usando las predicciones)
        pred_arrays = [pred for pred in preds_dict.values()]

        # ANOVA
        anova_result = f_oneway(*pred_arrays)

        # Tukey HSD
        all_preds = np.concatenate(pred_arrays)
        groups = np.concatenate([[name]*len(y_test) for name in preds_dict.keys()])

        tukey = pairwise_tukeyhsd(
            endog=all_preds,
            groups=groups,
            alpha=0.05
        )

        results['global'] = {
            'anova': {
                'statistic': anova_result.statistic,
                'p-value': anova_result.pvalue,
                'significativo': anova_result.pvalue < 0.05
            },
            'tukey': {
                'summary': tukey.summary(),
                'reject': tukey.reject,
                'meandiffs': tukey.meandiffs,
                'confint': tukey.confint,
                'q_crit': tukey.q_crit,
                'df_total': tukey.df_total
            }
        }

    return results

def generate_graphics_pdf(models_info, classes):
    """Genera un PDF solo con los gráficos de los modelos"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título del reporte
    title = Paragraph("Reporte Gráfico - Diagnóstico de Enfermedades en Hojas de Papa", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 24))

    # Agregar todos los gráficos
    for model_name, info in models_info.items():
        if model_name == 'statistical_tests':
            continue

        story.append(Paragraph(f"Modelo: {model_name}", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Matriz de confusión
        img_buffer = io.BytesIO()
        info['confusion_matrix'].savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        confusion_img = PlatypusImage(img_buffer, width=450, height=350)
        story.append(confusion_img)
        story.append(Spacer(1, 12))

        # Mapa de calor
        if 'heatmap' in info:
            heatmap_buffer = io.BytesIO()
            info['heatmap'].savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=300)
            heatmap_buffer.seek(0)
            heatmap_img = PlatypusImage(heatmap_buffer, width=500, height=150)
            story.append(heatmap_img)
            story.append(Spacer(1, 12))

        # Curvas de aprendizaje
        if info['learning_curves'] is not None:
            img_buffer2 = io.BytesIO()
            info['learning_curves'].savefig(img_buffer2, format='png', bbox_inches='tight', dpi=300)
            img_buffer2.seek(0)
            learning_img = PlatypusImage(img_buffer2, width=500, height=250)
            story.append(learning_img)
        story.append(Spacer(1, 24))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_text_analysis_pdf(models_info, classes):
    """Genera un PDF con el análisis textual de los resultados"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título del reporte
    title = Paragraph("Análisis de Resultados - Diagnóstico de Enfermedades en Hojas de Papa", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Resumen ejecutivo
    story.append(Paragraph("Resumen Ejecutivo", styles['Heading2']))
    story.append(Paragraph(
        "Este documento presenta un análisis detallado de los modelos de clasificación "
        "entrenados para identificar enfermedades en hojas de papa. Se incluyen interpretaciones "
        "de las métricas de rendimiento, matrices de confusión, mapas de calor y curvas de aprendizaje.",
        styles['Normal']))
    story.append(Spacer(1, 12))

    # Análisis por modelo
    for model_name, info in models_info.items():
        if model_name == 'statistical_tests':
            continue

        story.append(Paragraph(f"Análisis del Modelo: {model_name}", styles['Heading2']))

        # Métricas de rendimiento
        story.append(Paragraph("Métricas de Rendimiento:", styles['Heading3']))
        metrics_text = (
            f"El modelo {model_name} alcanzó una exactitud de {info['accuracy']:.2%}, "
            f"con una precisión de {info['precision']:.2%} y un recall de {info['recall']:.2%}. "
            f"El F1-Score de {info['f1']:.2%} y el coeficiente de correlación de Matthews (MCC) de "
            f"{info['mcc']:.4f} indican un rendimiento general del modelo."
        )
        story.append(Paragraph(metrics_text, styles['Normal']))
        story.append(Spacer(1, 8))

        # Interpretación matriz de confusión
        story.append(Paragraph("Interpretación de la Matriz de Confusión:", styles['Heading3']))
        conf_text = (
            "La matriz de confusión muestra cómo el modelo clasifica correcta e incorrectamente "
            "las diferentes clases. Las casillas en la diagonal principal representan las "
            "clasificaciones correctas, mientras que las otras casillas muestran los errores. "
            "Los patrones de error pueden revelar qué clases son más difíciles de distinguir."
        )
        story.append(Paragraph(conf_text, styles['Normal']))
        story.append(Spacer(1, 8))

        # Interpretación mapa de calor
        story.append(Paragraph("Interpretación del Mapa de Calor:", styles['Heading3']))
        heatmap_text = (
            "El mapa de calor de probabilidades muestra cómo el modelo distribuye su certeza entre las "
            "diferentes clases para cada predicción. Las celdas más oscuras indican mayor probabilidad "
            "asignada a esa clase. Un buen modelo mostrará alta probabilidad en la clase correcta "
            "y baja en las demás."
        )
        story.append(Paragraph(heatmap_text, styles['Normal']))
        story.append(Spacer(1, 8))

        # Interpretación curvas de aprendizaje
        if info['learning_curves'] is not None:
            story.append(Paragraph("Interpretación de las Curvas de Aprendizaje:", styles['Heading3']))
            learning_text = (
                "Las curvas de aprendizaje muestran la evolución de la precisión y la pérdida durante "
                "el entrenamiento. Una curva de entrenamiento que converge con la curva de validación "
                "indica un buen ajuste del modelo. Grandes diferencias pueden sugerir sobreajuste "
                "(overfitting) o subajuste (underfitting)."
            )
            story.append(Paragraph(learning_text, styles['Normal']))
        story.append(Spacer(1, 12))

    # Comparación entre modelos
    if len(models_info) > 1 and 'statistical_tests' in models_info:
        stats_info = models_info['statistical_tests']

        story.append(Paragraph("Comparación Estadística entre Modelos", styles['Heading2']))

        # ANOVA
        if 'global' in stats_info and stats_info['global']:
            anova = stats_info['global']['anova']
            story.append(Paragraph("Análisis de Varianza (ANOVA):", styles['Heading3']))
            anova_text = (
                f"El ANOVA realizado mostró un estadístico F de {anova['statistic']:.4f} con un valor p de "
                f"{anova['p-value']:.4f}, lo que indica {'diferencias estadísticamente significativas' if anova['significativo'] else 'no hay diferencias significativas'} "
                "entre los modelos en términos de su rendimiento."
            )
            story.append(Paragraph(anova_text, styles['Normal']))
            story.append(Spacer(1, 8))

            # Tukey
            if anova['significativo']:
                story.append(Paragraph("Prueba Post-Hoc de Tukey:", styles['Heading3']))
                tukey_text = (
                    "La prueba de Tukey identifica qué pares de modelos difieren significativamente. "
                    "Las comparaciones con 'Rechazar = Sí' indican diferencias estadísticamente significativas "
                    "entre esos modelos."
                )
                story.append(Paragraph(tukey_text, styles['Normal']))
                story.append(Spacer(1, 8))

        # McNemar
        if 'pairwise' in stats_info and stats_info['pairwise']:
            story.append(Paragraph("Pruebas de McNemar:", styles['Heading3']))
            mcnemar_text = (
                "La prueba de McNemar evalúa si dos modelos tienen proporciones de error diferentes. "
                "Un valor p significativo sugiere que un modelo comete significativamente más errores "
                "que el otro en las mismas muestras."
            )
            story.append(Paragraph(mcnemar_text, styles['Normal']))
            story.append(Spacer(1, 8))

    # Conclusiones y recomendaciones
    story.append(Paragraph("Conclusiones y Recomendaciones", styles['Heading2']))
    conclusions = [
        "1. El modelo con mejor rendimiento general debería ser seleccionado para implementación.",
        "2. Los patrones de error comunes pueden sugerir áreas para mejorar el conjunto de datos.",
        "3. Las diferencias significativas entre modelos justifican la selección del mejor clasificador.",
        "4. Para producción, considerar no solo la exactitud sino también la robustez del modelo.",
        "5. Monitorear continuamente el rendimiento del modelo con nuevos datos."
    ]

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_pdf_report(models_info, classes, test_images=None):
    """Genera un reporte PDF con los resultados"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título del reporte
    title = Paragraph("Reporte de Diagnóstico de Enfermedades en Hojas de Papa", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Resumen de modelos
    story.append(Paragraph("Resumen de Modelos", styles['Heading2']))

    # Tabla de métricas
    data = [['Modelo', 'Exactitud', 'Precisión', 'Recall', 'F1-Score', 'MCC']]
    for model_name, info in models_info.items():
        if model_name == 'statistical_tests':
            continue
        data.append([
            model_name,
            f"{info['accuracy']:.2%}",
            f"{info['precision']:.2%}",
            f"{info['recall']:.2%}",
            f"{info['f1']:.2%}",
            f"{info['mcc']:.4f}"
        ])

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Gráficos
    story.append(Paragraph("Gráficos de Rendimiento", styles['Heading2']))

    for model_name, info in models_info.items():
        if model_name == 'statistical_tests':
            continue

        # Matriz de confusión
        img_buffer = io.BytesIO()
        info['confusion_matrix'].savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        confusion_img = PlatypusImage(img_buffer, width=400, height=300)

        # Mapa de calor
        if 'heatmap' in info:
            heatmap_buffer = io.BytesIO()
            info['heatmap'].savefig(heatmap_buffer, format='png', bbox_inches='tight')
            heatmap_buffer.seek(0)
            heatmap_img = PlatypusImage(heatmap_buffer, width=500, height=150)

        # Curvas de aprendizaje
        if info['learning_curves'] is not None:
            img_buffer2 = io.BytesIO()
            info['learning_curves'].savefig(img_buffer2, format='png', bbox_inches='tight')
            img_buffer2.seek(0)
            learning_img = PlatypusImage(img_buffer2, width=500, height=250)

        story.append(Paragraph(f"Modelo: {model_name}", styles['Heading3']))
        story.append(confusion_img)
        story.append(Spacer(1, 12))
        if 'heatmap' in info:
            story.append(heatmap_img)
            story.append(Spacer(1, 12))
        if info['learning_curves'] is not None:
            story.append(learning_img)
        story.append(Spacer(1, 24))

    # Pruebas estadísticas si hay más de un modelo
    if 'statistical_tests' in models_info:
        stats_info = models_info['statistical_tests']

        # ANOVA
        if 'global' in stats_info and stats_info['global']:
            story.append(Paragraph("Análisis Estadístico Global", styles['Heading2']))

            anova = stats_info['global']['anova']
            story.append(Paragraph(f"ANOVA de una vía:", styles['Heading3']))
            story.append(Paragraph(f"Estadístico F: {anova['statistic']:.4f}", styles['Normal']))
            story.append(Paragraph(f"Valor p: {anova['p-value']:.4f} {'(Significativo)' if anova['significativo'] else '(No significativo)'}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Tukey
            if anova['significativo']:
                story.append(Paragraph("Prueba Post-Hoc de Tukey:", styles['Heading3']))

                tukey_data = [
                    ['Grupo 1', 'Grupo 2', 'Diferencia', 'Inferior', 'Superior', 'Rechazar']
                ]

                tukey = stats_info['global']['tukey']
                for i in range(len(tukey['reject'])):
                    tukey_data.append([
                        tukey['summary'].data[i+1][0],
                        tukey['summary'].data[i+1][1],
                        f"{tukey['meandiffs'][i]:.4f}",
                        f"{tukey['confint'][i][0]:.4f}",
                        f"{tukey['confint'][i][1]:.4f}",
                        'Sí' if tukey['reject'][i] else 'No'
                    ])

                t = Table(tukey_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

        # Comparaciones por pares
        if 'pairwise' in stats_info and stats_info['pairwise']:
            story.append(Paragraph("Comparaciones por Pares", styles['Heading2']))

            # McNemar
            story.append(Paragraph("Prueba de McNemar:", styles['Heading3']))
            mcnemar_data = [['Modelos', 'Estadístico', 'Valor p', 'Significativo']]
            for comp, tests in stats_info['pairwise'].items():
                mcnemar_data.append([
                    comp,
                    f"{tests['mcnemar']['statistic']:.4f}",
                    f"{tests['mcnemar']['p-value']:.4f}",
                    'Sí' if tests['mcnemar']['significativo'] else 'No'
                ])

            t = Table(mcnemar_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # t-test
            story.append(Paragraph("Prueba t pareada (exactitud):", styles['Heading3']))
            ttest_data = [['Modelos', 'Estadístico t', 'Valor p', 'Significativo']]
            for comp, tests in stats_info['pairwise'].items():
                ttest_data.append([
                    comp,
                    f"{tests['t-test']['statistic']:.4f}",
                    f"{tests['t-test']['p-value']:.4f}",
                    'Sí' if tests['t-test']['significativo'] else 'No'
                ])

            t = Table(ttest_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

    # Recomendaciones generales
    story.append(Paragraph("Recomendaciones Generales", styles['Heading2']))
    recommendations = [
        "1. Para hojas saludables: Mantener un programa de fertilización balanceado y monitorear regularmente.",
        "2. Para Early Blight: Aplicar fungicidas preventivos y rotar cultivos.",
        "3. Para Late Blight: Usar fungicidas sistémicos y eliminar plantas infectadas.",
        "4. Siempre verificar diagnósticos con un agrónomo certificado.",
        "5. Tomar fotografías con buena iluminación y fondo uniforme para mejores resultados."
    ]

    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def get_recommendations(diagnosis, confidence):
    """Devuelve recomendaciones basadas en el diagnóstico con iconos"""
    specific_recs = {
        "Potato___Early_blight": [
            ("🔍", "Realizar inspección visual para confirmar manchas concéntricas características"),
            ("🔄", "Rotar cultivos para evitar acumulación del patógeno en el suelo"),
            ("🍃", "Aplicar fungicidas protectantes como clorotalonil o mancozeb"),
            ("💧", "Evitar riego por aspersión para reducir humedad foliar"),
            ("🧹", "Eliminar y destruir residuos de cultivos infectados")
        ],
        "Potato___Late_blight": [
            ("⚠️", "Aislamiento inmediato de plantas infectadas para prevenir propagación"),
            ("🔄", "Usar fungicidas sistémicos como metalaxil o fosetil-aluminio"),
            ("🌧️", "Monitorear condiciones climáticas (alta humedad favorece la enfermedad)"),
            ("🚜", "Destrucción completa de plantas gravemente afectadas"),
            ("📆", "Implementar programa estricto de aplicación de fungicidas")
        ],
        "Potato___healthy": [
            ("👍", "Continuar con buenas prácticas agrícolas actuales"),
            ("🔍", "Mantener programa de monitoreo semanal para detección temprana"),
            ("🌱", "Asegurar nutrición balanceada y riego adecuado"),
            ("🛡️", "Considerar aplicación preventiva de biocontroladores"),
            ("📝", "Documentar condiciones de crecimiento para referencia futura")
        ]
    }

    general_tips = [
        ("📸", "Para mejores resultados: Fotografíe la hoja con luz natural, fondo uniforme y enfoque claro"),
        ("🔄", "Validar diagnóstico automático con inspección física cuando sea posible"),
        ("📅", "Mantener registros históricos de diagnósticos para identificar patrones"),
        ("🌦️", "Considerar condiciones climáticas recientes en el análisis"),
        ("👨‍🌾", "Consultar a un especialista para casos complejos o de alta importancia económica")
    ]

    # Seleccionar recomendaciones específicas
    main_rec = specific_recs.get(diagnosis, [])

    # Ajustar según confianza
    if confidence < 0.7:
        confidence_rec = [("⚠️", f"Confianza moderada ({confidence:.0%}) - Se recomienda verificación adicional")]
    elif confidence > 0.9:
        confidence_rec = [("✅", f"Alta confianza ({confidence:.0%}) - Puede proceder con las recomendaciones")]
    else:
        confidence_rec = [("ℹ️", f"Confianza moderada ({confidence:.0%}) - Considere validación adicional")]

    # Formatear recomendaciones con iconos
    formatted_recs = [
        f"{icon} {text}".replace('\n', ' ')  # Eliminar saltos de línea
        for icon, text in confidence_rec + main_rec + general_tips
    ]

    return formatted_recs

def display_recommendations(diagnosis, confidence):
    """Muestra recomendaciones visualmente atractivas"""
    recommendations = get_recommendations(diagnosis, confidence)

    st.markdown('<div class="recommendation">', unsafe_allow_html=True)
    st.markdown('<p class="recommendation-header">📌 Recomendaciones para el diagnóstico:</p>', unsafe_allow_html=True)

    for rec in recommendations:
        st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def generate_diagnosis_pdf(model_name, diagnosis, confidence, prob_df, image, classes):
    """Genera un PDF específico para un diagnóstico individual"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título del reporte
    title = Paragraph(f"Reporte de Diagnóstico - {model_name}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Información del diagnóstico
    story.append(Paragraph(f"Diagnóstico: {diagnosis}", styles['Heading2']))
    story.append(Paragraph(f"Confianza: {confidence:.2%}", styles['Heading3']))
    story.append(Spacer(1, 12))

    # Imagen de la hoja (reducida para el PDF)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=70)
    img_buffer.seek(0)
    pdf_img = PlatypusImage(img_buffer, width=200, height=200)
    story.append(pdf_img)
    story.append(Spacer(1, 12))

    # Tabla de probabilidades
    prob_data = [['Clase', 'Probabilidad']]
    for _, row in prob_df.iterrows():
        prob_data.append([row['Clase'], f"{row['Probabilidad']:.2%}"])

    t = Table(prob_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Recomendaciones
    story.append(Paragraph("Recomendaciones Específicas", styles['Heading2']))
    recommendations = get_recommendations(diagnosis, confidence)
    for rec in recommendations[:10]:  # Mostrar hasta 10 recomendaciones
        # Eliminar saltos de línea y unir texto correctamente
        if isinstance(rec, str):
            clean_rec = rec.replace('\n', ' ').replace(' - ', ' ')
            story.append(Paragraph(clean_rec, styles['Normal']))
        elif isinstance(rec, (list, tuple)):
            clean_rec = ' '.join(str(x) for x in rec).replace('\n', ' ')
            story.append(Paragraph(clean_rec, styles['Normal']))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def display_statistical_results(results, classes):
    """Muestra los resultados estadísticos de forma organizada"""
    st.subheader("📊 Análisis Estadístico Comparativo")

    # 1. Resultados globales (ANOVA)
    if 'global' in results and results['global']:
        anova = results['global']['anova']
        tukey = results['global']['tukey']

        st.markdown('<div class="anova-result">', unsafe_allow_html=True)
        st.markdown("#### ANOVA de una vía")
        st.write(f"**Estadístico F:** {anova['statistic']:.4f}")
        st.write(f"**Valor p:** {anova['p-value']:.4f} {'(Significativo)' if anova['significativo'] else '(No significativo)'}")

        if anova['significativo']:
            st.markdown("#### Prueba Post-Hoc de Tukey")
            st.write("Diferencias significativas entre grupos:")

            # Crear DataFrame para Tukey
            tukey_data = []
            for i in range(len(tukey['reject'])):
                tukey_data.append({
                    'Grupo 1': tukey['summary'].data[i+1][0],
                    'Grupo 2': tukey['summary'].data[i+1][1],
                    'Diferencia': f"{tukey['meandiffs'][i]:.4f}",
                    'Límite inferior': f"{tukey['confint'][i][0]:.4f}",
                    'Límite superior': f"{tukey['confint'][i][1]:.4f}",
                    'Significativo': tukey['reject'][i]
                })

            tukey_df = pd.DataFrame(tukey_data)

            # Aplicar estilos condicionales
            def highlight_significant(row):
                return ['background-color: #c8e6c9' if row['Significativo'] else '' for _ in row]

            st.dataframe(tukey_df.style.apply(highlight_significant, axis=1))
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Comparaciones por pares
    if 'pairwise' in results and results['pairwise']:
        st.markdown("#### Comparaciones por Pares")

        # Tabla de McNemar
        mcnemar_data = []
        for comparison, tests in results['pairwise'].items():
            mcnemar_data.append({
                'Modelos': comparison,
                'Estadístico': tests['mcnemar']['statistic'],
                'Valor p': tests['mcnemar']['p-value'],
                'Significativo': 'Sí' if tests['mcnemar']['significativo'] else 'No'
            })

        st.markdown("**Prueba de McNemar**")
        st.dataframe(pd.DataFrame(mcnemar_data))

        # Tabla de t-test pareado
        ttest_data = []
        for comparison, tests in results['pairwise'].items():
            ttest_data.append({
                'Modelos': comparison,
                'Estadístico t': tests['t-test']['statistic'],
                'Valor p': tests['t-test']['p-value'],
                'Significativo': 'Sí' if tests['t-test']['significativo'] else 'No'
            })

        st.markdown("**Prueba t pareada (exactitud)**")
        st.dataframe(pd.DataFrame(ttest_data))

# =============================================
# INTERFAZ PRINCIPAL
# =============================================

# Cargar datos
X_train, X_test, y_train, y_test, classes = load_data(dataset_path)

if X_train is not None:
    # Mostrar información del dataset
    st.subheader("📊 Información del Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Muestras de entrenamiento:** {len(X_train)}")
        st.info(f"**Muestras de validación:** {len(X_test)}")
        st.info(f"**Clases:** {len(classes)}")

    with col2:
        # Distribución de clases
        fig, ax = plt.subplots(figsize=(8, 4))
        class_dist = np.sum(y_train, axis=0)
        custom_colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0', '#F0E68C', '#FFB6C1', '#87CEFA']
        bar_colors = custom_colors * (len(classes) // len(custom_colors) + 1)
        ax.bar(classes, class_dist, color=bar_colors[:len(classes)])
        ax.set_title("Distribución de Clases")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Ejemplos de imágenes
    st.subheader("🌿 Ejemplos de Imágenes")
    num_examples = 3
    image_size = 3
    fig, axes = plt.subplots(len(classes), num_examples, figsize=(image_size * num_examples, image_size * len(classes)))

    for i, class_name in enumerate(classes):
        class_images = X_train[np.argmax(y_train, axis=1) == i]
        for j in range(num_examples):
            ax = axes[i,j]
            if j < len(class_images):
                ax.imshow(class_images[j])
                ax.axis('off')
                if j == num_examples//2:
                    ax.set_title(class_name, fontsize=14)
            else:
                ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # Entrenamiento de modelos
    if st.button("🚀 Entrenar Modelos", type="primary"):
        if not selected_models:
            st.warning("Por favor selecciona al menos un modelo")
        else:
            with st.spinner("Entrenando modelos..."):
                results = {}
                preds_dict = {}  # Para almacenar predicciones para pruebas estadísticas

                for model_name in selected_models:
                    with st.expander(f"Modelo: {model_name}", expanded=True):
                        model = None
                        history = None

                        # Intentar cargar modelo guardado si está habilitado
                        if load_saved_models:
                            model_path = f"/content/drive/MyDrive/potato_model_{model_name}.h5"
                            try:
                                model = load_model(model_path)
                                st.success(f"✅ Modelo {model_name} cargado desde {model_path}")
                            except Exception as e:
                                st.warning(f"⚠️ No se pudo cargar el modelo {model_name}, entrenando desde cero: {str(e)}")

                        # Si no se cargó el modelo, entrenar desde cero
                        if model is None:
                            model = create_model(model_name, len(classes), learning_rate)
                            model, history = train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size)
                            st.success(f"✅ Entrenamiento completado para {model_name}")

                        # Evaluar modelo
                        y_pred = model.predict(X_test)
                        y_pred_classes = np.argmax(y_pred, axis=1)
                        y_test_classes = np.argmax(y_test, axis=1)

                        # Calcular métricas
                        accuracy = accuracy_score(y_test_classes, y_pred_classes)
                        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
                        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
                        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
                        mcc = matthews_corrcoef(y_test_classes, y_pred_classes)

                        # Generar visualizaciones
                        cm_fig = plot_confusion_matrix(y_test_classes, y_pred_classes, classes)
                        lc_fig = plot_learning_curves(history) if history is not None else None

                        # Mostrar resultados
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Exactitud", f"{accuracy:.2%}")
                            st.metric("Precisión", f"{precision:.2%}")
                            st.metric("MCC", f"{mcc:.4f}")
                        with col2:
                            st.metric("Recall", f"{recall:.2%}")
                            st.metric("F1-Score", f"{f1:.2%}")

                        # Mostrar visualizaciones
                        st.pyplot(cm_fig)
                        if lc_fig is not None:
                            st.pyplot(lc_fig)

                        # Guardar resultados
                        results[model_name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'mcc': mcc,
                            'confusion_matrix': cm_fig,
                            'learning_curves': lc_fig,
                            'history': history
                        }

                        # Guardar predicciones para pruebas estadísticas
                        preds_dict[model_name] = y_pred_classes

                        # Guardar modelo en Drive si fue entrenado
                        if history is not None:
                            try:
                                model_path = os.path.join("/content/drive/MyDrive", f"potato_model_{model_name}.h5")
                                model.save(model_path)
                                st.success(f"Modelo guardado en: {model_path}")
                            except Exception as e:
                                st.error(f"Error al guardar modelo: {str(e)}")

                # Realizar pruebas estadísticas si hay al menos 2 modelos
                if len(selected_models) >= 2:
                    stats_results = statistical_tests(y_test_classes, preds_dict)
                    results['statistical_tests'] = stats_results

                    # Mostrar resultados estadísticos
                    display_statistical_results(stats_results, classes)

                st.session_state.results = results
                st.balloons()

                # Generar PDFs con los resultados
                if len(selected_models) > 0:
                    # PDF de gráficos
                    graphics_pdf = generate_graphics_pdf(results, classes)
                    # PDF de análisis de texto
                    text_pdf = generate_text_analysis_pdf(results, classes)
                    # PDF combinado (original)
                    combined_pdf = generate_pdf_report(results, classes)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Descargar Reporte Gráfico (PDF)",
                            data=graphics_pdf,
                            file_name="diagnostico_papa_graficos.pdf",
                            mime="application/pdf",
                            key="graphics_report"
                        )
                    with col2:
                        st.download_button(
                            label="📥 Descargar Análisis (PDF)",
                            data=text_pdf,
                            file_name="diagnostico_papa_analisis.pdf",
                            mime="application/pdf",
                            key="text_report"
                        )
                    with col3:
                        st.download_button(
                            label="📥 Descargar Reporte Completo (PDF)",
                            data=combined_pdf,
                            file_name="diagnostico_papa_completo.pdf",
                            mime="application/pdf",
                            key="full_report"
                        )

# Mostrar resultados si están disponibles
if 'results' in st.session_state and st.session_state.results:
    st.subheader("📈 Resultados Comparativos")

    # Eliminar tests estadísticos para la tabla (si existen)
    display_results = {k: v for k, v in st.session_state.results.items() if k != 'statistical_tests'}

    # Tabla comparativa
    results_df = pd.DataFrame.from_dict({
        model: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'MCC': metrics['mcc']
        }
        for model, metrics in display_results.items()
    }, orient='index')

    st.dataframe(results_df.style.format("{:.2%}").background_gradient(cmap='Blues'))

    # Gráfico comparativo
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind='bar', ax=ax)
    ax.set_title("Comparación de Modelos")
    ax.set_ylabel("Puntuación")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sección para probar con nuevas imágenes
st.subheader("🔍 Probar con Nueva Imagen")
uploaded_file = st.file_uploader("Sube una imagen de hoja de papa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and 'results' in st.session_state:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", width=300)

    # Preprocesar imagen
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Hacer predicciones
    st.subheader("📋 Resultados del Diagnóstico")

    for model_name, metrics in st.session_state.results.items():
        if model_name == 'statistical_tests':
            continue

        model = metrics['model']
        pred = model.predict(image_array)[0]

        with st.expander(f"Predicciones - {model_name}", expanded=True):
            # Mostrar probabilidades
            prob_df = pd.DataFrame({
                'Clase': classes,
                'Probabilidad': pred
            }).sort_values('Probabilidad', ascending=False)

            st.dataframe(prob_df.style.format({'Probabilidad': '{:.2%}'}).bar(subset=['Probabilidad'], color='#5fba7d'))

            # Mostrar mapa de calor
            st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
            st.subheader("Mapa de Calor de Probabilidades")
            heatmap_fig = plot_heatmap(pred, classes, model_name)
            st.pyplot(heatmap_fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Guardar heatmap para el PDF
            st.session_state.results[model_name]['heatmap'] = heatmap_fig

            # Mostrar diagnóstico principal
            top_pred = prob_df.iloc[0]
            st.success(f"Diagnóstico: {top_pred['Clase']} ({top_pred['Probabilidad']:.2%} de confianza)")

            # Mostrar recomendaciones mejoradas
            display_recommendations(top_pred['Clase'], top_pred['Probabilidad'])

            # Generar PDF con el diagnóstico específico
            diagnosis_buffer = generate_diagnosis_pdf(
                model_name,
                top_pred['Clase'],
                top_pred['Probabilidad'],
                prob_df,
                image,
                classes
            )

            st.download_button(
                label=f"📥 Descargar Diagnóstico {model_name} (PDF)",
                data=diagnosis_buffer,
                file_name=f"diagnostico_{model_name}.pdf",
                mime="application/pdf",
                key=f"download_{model_name}"
            )