'''
#########################################################################################################################
INFORMACIÓN DEL FICHERO
#########################################################################################################################

Autores:     Stefany Marisel Pérez Bazán · José Andrés Farro Lagos
Institución: Universidad Nacional de Trujillo (UNT)
Asesor:      Dr. Juan Pedro Santos Fernández
Fecha:       2025-09-20

Descripción (ES):
    Aplicación web para entrenar, comparar y utilizar modelos de Deep Learning (CNN + Transfer Learning)
    para diagnosticar enfermedades en hojas de papa: Hoja sana, Tizón temprano (Early blight),
    Tizón tardío (Late blight). Incluye métricas, estadísticas (ANOVA, McNemar, Tukey) y exportación de
    reportes PDF (técnico, entrenamiento, gráficas, interpretación, diagnóstico).

Description (EN):
    Web application to train, compare and use Deep Learning models (CNN + Transfer Learning) to diagnose
    potato leaf diseases: Healthy, Early blight, Late blight. It includes metrics, statistics (ANOVA,
    McNemar, Tukey) and PDF exports (technical, training, plots, interpretation, diagnosis).

Características clave / Key features:
    • Modelos: Custom CNN, MobileNetV2, ResNet50V2, Xception, DenseNet121
    • Métricas: Accuracy, Precision, Recall, F1, MCC · Gráficos: Curvas de aprendizaje, CM, ROC
    • Diagnóstico individual con recomendaciones según enfermedad y confianza
    • Multidioma (Español / English) en UI y reportes
    • Guardado/Carga de modelos con metadatos (épocas, curvas, clases)

Dataset:
    Ruta recomendada en Colab: /content/drive/MyDrive/Colab_Data/potato_data/PlantVillage
    Estructura: {class}/imagen.jpg con clases: Potato___healthy, Potato___Early_blight, Potato___Late_blight

Uso rápido (Colab):
    1) Activa GPU · 2) Monta Drive · 3) Instala dependencias · 4) run streamlit
    !pip -q install tensorflow keras streamlit scikit-learn statsmodels reportlab matplotlib pillow localtunnel
    !streamlit run /content/potato_disease_diagnosis.py & npx localtunnel --port 8501

Notas:
    - No se muestran resultados pesados hasta que el usuario presiona “Show Results”.
    - Si no seleccionas modelos y no habilitas carga, se muestra un aviso (no se ejecuta nada).
    - Al cargar modelos, las curvas/épocas se restauran desde metadatos cuando estén disponibles.

#########################################################################################################################
'''

import os, io, glob, json, datetime
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ML / Métricas / Estadística
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    confusion_matrix, roc_curve, auc
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison

# PDFs / gráficas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, LongTable, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
import matplotlib.pyplot as plt

# ---------- i18n ----------
LANG_LABELS = {"es": "Español", "en": "English"}
DEFAULT_LANG = "es"

# Base de claves en INGLÉS. ES traduce todo a español.
ES = {
    # Secciones / tabs
    "System":"Sistema","Train":"Entrenamiento","Reports":"Reportes","Diagnosis":"Diagnóstico","Help & About":"Ayuda y Acerca de",
    # Sidebar
    "Language":"Idioma","Load/Save":"Cargar/Guardar","Parameters":"Parámetros","Models":"Modelos",
    "Load Saved Models":"Cargar modelos guardados","Save Trained Models":"Guardar modelos entrenados",
    "Dataset path":"Ruta del dataset","Validation split":"Proporción de validación","Epochs":"Épocas","Batch size":"Tamaño de lote","Learning rate":"Tasa de aprendizaje",
    # Sistema
    "Software Specifications":"Especificaciones de software","Library Versions":"Versiones de librerías",
    "Requirements":"Requisitos","Version":"Versión","Library":"Librería",
    "Operating System":"Sistema Operativo","Python Version":"Versión de Python","CPU":"CPU","RAM":"RAM","GPU":"GPU",
    # Botones / acciones
    "Train Models":"Entrenar modelos","Run Stats":"Ejecutar estadísticas",
    "Load saved models now":"Cargar modelos guardados ahora",
    "Show Results":"Mostrar resultados","Hide Results":"Ocultar resultados",
    # Mensajes
    "Nothing to show yet. Select models or enable loading and press the load button.":"Nada que mostrar aún. Selecciona modelos o habilita carga y presiona el botón de cargar.",
    "No dataset found":"No se encontró el dataset","Training finished":"Entrenamiento finalizado",
    "Learning curves note":"Nota: si cargas modelos guardados, se restaurarán curvas/épocas desde metadatos (si existen).",
    "No saved models. Please train models first.":"No hay modelos guardados. Primero entrena los modelos.",
    "No models loaded":"No hay modelos cargados. Carga o entrena antes de diagnosticar.",
    "Selected models":"Modelos seleccionados",
    "Models loaded. Press 'Show Results' to display results.":"Modelos cargados. Presiona 'Mostrar resultados' para verlos.",
    "You have not selected any model nor enabled loading saved models.":"No has seleccionado modelos ni activado la carga de modelos guardados.",
    # Métricas y gráficas
    "Learning Curves":"Curvas de aprendizaje","Accuracy":"Exactitud","Loss":"Pérdida","Precision":"Precisión","Recall":"Sensibilidad","F1-Score":"F1-Score","MCC":"Coeficiente de Matthews",
    "Confusion Matrix":"Matriz de confusión","ROC Curves":"Curvas ROC","Classes":"Clases",
    "Confusion Matrix (Validation)":"Matriz de confusión (Validación)","ROC Curves (Validation)":"Curvas ROC (Validación)",
    "Learning Curves (Accuracy/Loss)":"Curvas de aprendizaje (Exactitud/Pérdida)","Epochs used":"Épocas usadas",
    # Dataset info
    "Dataset info":"Información del dataset","Training samples":"Muestras de entrenamiento","Validation samples":"Muestras de validación","Class":"Clase","Total":"Total","% Total":"% Total","% Train":"% Entrenamiento","% Val":"% Validación","Dataset Samples":"Muestras del dataset",
    # Reportes
    "Technical Report":"Reporte Técnico","Training Report":"Reporte de Entrenamiento","Diagnosis Report":"Reporte de Diagnóstico","Graphs Report":"Reporte de Gráficas","Interpretation Report":"Reporte de Interpretación",
    "Generate Technical PDF":"Generar PDF Técnico","Generate Training PDF":"Generar PDF de Entrenamiento","Generate Diagnosis PDF":"Generar PDF de Diagnóstico","Generate Graphs PDF":"Generar PDF de Gráficas","Generate Interpretation PDF":"Generar PDF de Interpretación",
    "Download Technical PDF":"Descargar PDF Técnico","Download Training PDF":"Descargar PDF de Entrenamiento","Download Diagnosis PDF":"Descargar PDF de Diagnóstico","Download Graphs PDF":"Descargar PDF de Gráficas","Download Interpretation PDF":"Descargar PDF de Interpretación",
    "Summary":"Resumen","Created at":"Creado el","Curves source (trained)":"Fuente de curvas: Entrenamiento actual","Curves source (loaded)":"Fuente de curvas: Metadatos (carga)",
    # Interpretación
    "Possible overfitting (large train/val gap and/or rising val loss).":"Posible sobreajuste (brecha train/val grande y/o pérdida de validación en aumento).",
    "Possible underfitting (low accuracies and/or high losses).":"Posible subajuste (exactitudes bajas y/o pérdidas altas).",
    "Good fit (high accuracies, stable curves, small gap).":"Ajuste adecuado (exactitudes altas, curvas estables, brecha pequeña).",
    "ROC: closer to top-left is better; AUC summarizes per-class performance.":"ROC: cuanto más arriba-izquierda, mejor; el AUC resume el rendimiento por clase.",
    "Confusion Matrix: high diagonal means good performance; inspect off-diagonal confusions.":"Matriz de confusión: diagonal alta = buen desempeño; observa confusiones fuera de la diagonal.",
    "Learning Curves: compare train vs val; look for convergence without divergence.":"Curvas de aprendizaje: compara entrenamiento vs validación; busca convergencia sin divergencia.",
    # Estadísticas
    "Statistics":"Estadísticas","ANOVA":"ANOVA","ANOVA Summary":"Resumen ANOVA","McNemar (pairs)":"McNemar (pares)","Tukey HSD":"Tukey HSD",
    "Model A":"Modelo A","Model B":"Modelo B","F-statistic":"Estadístico F","p-value":"p-valor","Significant (α=0.05)":"Significativo (α=0.05)","Mean Diff":"Diferencia media","Lower CI":"Límite inferior","Upper CI":"Límite superior","Reject H0":"Rechaza H₀","N01":"N01","N10":"N10",
    "Stats need at least 2 models":"Las pruebas estadísticas requieren al menos 2 modelos","Execution canceled: select more models.":"Ejecución cancelada: selecciona más modelos.",
    # Diagnóstico
    "Upload image":"Subir imagen","Predict":"Diagnosticar","Model for diagnosis":"Modelo para diagnóstico","Per-class probabilities":"Probabilidades por clase","Diagnosis result":"Resultado del diagnóstico","Top prediction":"Mejor predicción","Confidence":"Confianza","Recommendation":"Recomendación",
    # Manual (EN→ES)
    "User Manual":"Manual de usuario",
    "A. Parameters":"A. Parámetros",
    "Dataset path:":"Ruta del dataset:",
    "Validation split / Epochs / Batch size / Learning rate.":"Proporción de validación / Épocas / Tamaño de lote / Tasa de aprendizaje.",
    "B. Load/Save":"B. Cargar/Guardar",
    "Load Saved Models: check the box, then press 📦 Load saved models now.":"Cargar modelos guardados: marca la casilla y luego pulsa 📦 Cargar modelos guardados ahora.",
    "Save Trained Models: save weights and metadata after training.":"Guardar modelos entrenados: guarda pesos y metadatos tras el entrenamiento.",
    "Learning curves note (loaded models may restore curves/epochs from metadata).":"Nota: los modelos cargados pueden restaurar curvas/épocas desde metadatos.",
    "C. Recommended flow":"C. Flujo recomendado",
    "Select Models.":"Selecciona Modelos.",
    "Press 🚀 Train Models (or enable loading and press 📦).":"Pulsa 🚀 Entrenar modelos (o habilita carga y pulsa 📦).",
    "Then press 👁️ Show Results to see curves, CM, ROC and metrics.":"Luego pulsa 👁️ Mostrar resultados para ver curvas, CM, ROC y métricas.",
    "Use 📊 Run Stats to compare.":"Usa 📊 Ejecutar estadísticas para comparar.",
    "Generate PDFs from Reports.":"Genera PDFs desde Reportes.",
    "D. Diagnosis":"D. Diagnóstico",
    "Single-model diagnosis.":"Diagnóstico individual por modelo.",
    "You will see Diagnosis result, per-class table and recommendations tailored by disease and confidence.":"Verás Resultado del diagnóstico, tabla por clase y recomendaciones según enfermedad y confianza.",
    "The PDF includes the analyzed image.":"El PDF incluye la imagen analizada.",
    # Advertencias flujo
    "No models selected and loading not enabled.":"No se han seleccionado modelos y no se activó la carga de modelos.",
}

def t(s: str) -> str:
    """Traduce claves (base EN). En EN devuelve la clave; en ES traduce."""
    return ES.get(s, s) if st.session_state.get("lang", DEFAULT_LANG) == "es" else s

def clean_heading(s):
    return (s or "").replace("_"," ").replace("-"," ").strip()

# ---------- Estado ----------
if "models" not in st.session_state: st.session_state.models={}
if "results" not in st.session_state: st.session_state.results={}
if "pdfs" not in st.session_state: st.session_state.pdfs={}
if "trained_once" not in st.session_state: st.session_state.trained_once=False
if "lang" not in st.session_state: st.session_state.lang=DEFAULT_LANG
if "show_models" not in st.session_state: st.session_state.show_models=False
if "last_img_bytes" not in st.session_state: st.session_state.last_img_bytes=None

# ---------- Config/UI base ----------
st.set_page_config(page_title="Sistema de Diagnóstico de Enfermedades en Hojas de Papa con Deep Learning", layout="wide")
st.sidebar.title("⚙️")
st.session_state.lang = st.sidebar.radio(
    t("Language"), ["es","en"],
    index=0 if st.session_state.lang=="es" else 1,
    format_func=lambda c: LANG_LABELS[c], horizontal=True, key="lang_selector"
)

# --- Sidebar (sin defaults marcados) ---
st.sidebar.subheader(t("Load/Save"))
load_saved = st.sidebar.checkbox(t("Load Saved Models"), value=False, key="load_models")
save_trained = st.sidebar.checkbox(t("Save Trained Models"), value=False, key="save_models")

st.sidebar.subheader(t("Parameters"))
default_dataset = "/content/drive/MyDrive/Colab_Data/potato_data/PlantVillage"
dataset_path = st.sidebar.text_input(t("Dataset path"), value=default_dataset, key="dataset_path")
val_split = st.sidebar.slider(t("Validation split"), 0.05, 0.4, 0.2, 0.05, key="val_split")
epochs = st.sidebar.slider(t("Epochs"), 1, 20, 3, key="epochs")
batch_size = st.sidebar.selectbox(t("Batch size"), [8,16,32,64], index=2, key="batch_size")
learning_rate = st.sidebar.selectbox(t("Learning rate"), [1e-4,2e-4,5e-4,1e-3], index=3, key="learning_rate")

MODEL_KEYS = ["custom_cnn","mobilenetv2","resnet50v2","xception","densenet121"]
MODEL_LABELS = {
    "custom_cnn":{"es":"CNN Personalizada","en":"Custom CNN"},
    "mobilenetv2":{"es":"MobileNetV2","en":"MobileNetV2"},
    "resnet50v2":{"es":"ResNet50V2","en":"ResNet50V2"},
    "xception":{"es":"Xception","en":"Xception"},
    "densenet121":{"es":"DenseNet121","en":"DenseNet121"},
}
def model_display(k): return MODEL_LABELS[k][st.session_state.lang]

selected_models = st.sidebar.multiselect(
    t("Models"), MODEL_KEYS, default=[], format_func=model_display, key="sel_models"
)

st.title("🧪 Sistema de Diagnóstico de Enfermedades en Hojas de Papa con Deep Learning")
tabs = st.tabs([f"🖥️ {t('System')}", f"🧬 {t('Train')}", f"📊 {t('Reports')}", f"🩺 {t('Diagnosis')}", f"❓ {t('Help & About')}"])

# ---------- Sistema / Especificaciones y versiones ----------
def get_software_specs_df(lang):
    try:
        import psutil; ram=f"{round(psutil.virtual_memory().total/(1024**3),1)} GB"
    except Exception:
        ram="—"
    gpus=tf.config.list_physical_devices('GPU'); gpu=gpus[0].name if gpus else "—"
    data=[[t("Operating System"), platform.platform()],
          [t("Python Version"), platform.python_version()],
          [t("CPU"), platform.processor() or "—"],
          [t("RAM"), ram],[t("GPU"), gpu]]
    df = pd.DataFrame(data, columns=[t("Requirements"), t("Version")])
    return df

def get_library_versions_df(lang):
    versions = [
        ("TensorFlow", tf.__version__),
        ("Keras", keras.__version__),
        ("Streamlit", st.__version__),
        ("NumPy", np.__version__),
        ("Pandas", pd.__version__),
        ("Pillow", Image.__version__ if hasattr(Image,"__version__") else "—"),
        ("scikit-learn", __import__("sklearn").__version__),
        ("Statsmodels", __import__("statsmodels").__version__),
        ("SciPy", __import__("scipy").__version__),
        ("ReportLab", __import__("reportlab").Version),
        ("Matplotlib", matplotlib.__version__),
    ]
    df = pd.DataFrame(versions, columns=[t("Library"), t("Version")])
    return df

with tabs[0]:
    st.subheader(t("Software Specifications"))
    st.dataframe(get_software_specs_df(st.session_state.lang), use_container_width=True, hide_index=True)
    st.subheader(t("Library Versions"))
    st.dataframe(get_library_versions_df(st.session_state.lang), use_container_width=True, hide_index=True)

    # Overview dataset (ligero; solo recuento de archivos)
    class_dirs=[p.name for p in Path(dataset_path).glob("*") if p.is_dir()]
    if class_dirs:
        def dataset_overview_quick(root, val_split, class_names):
            data={}
            for c in class_names:
                count=len(glob.glob(os.path.join(root,c,"*")))
                data[c]=count
            total=sum(data.values()) or 1
            df=pd.DataFrame({t("Class"):list(data.keys()),t("Total"):[data[c] for c in data]})
            df[t("% Total")]=df[t("Total")]/total*100
            df[t("Training samples")]= (df[t("Total")]*(1-val_split)).astype(int)
            df[t("Validation samples")]=df[t("Total")]-df[t("Training samples")]
            df[t("% Train")]=df[t("Training samples")]/df[t("Total")].clip(lower=1)*100
            df[t("% Val")]=100-df[t("% Train")]
            return df
        info=dataset_overview_quick(dataset_path, val_split, class_dirs)
        st.subheader(t("Dataset info"))
        st.dataframe(info, use_container_width=True, hide_index=True)
        fig=plt.figure(); plt.bar(info[t("Class")], info[t("Total")]); plt.xticks(rotation=45,ha="right"); plt.ylabel(t("Total")); st.pyplot(fig)
        fig2=plt.figure(); plt.pie(info[t("% Total")], labels=info[t("Class")], autopct="%1.1f%%"); st.pyplot(fig2)

# ---------- Datos / utilidades ----------
def load_data(dirpath, img_size=(224,224), vsplit=0.2, bs=32, seed=42):
    if not os.path.isdir(dirpath): return None, None, []
    train=tf.keras.preprocessing.image_dataset_from_directory(
        dirpath, validation_split=vsplit, subset="training", seed=seed,
        image_size=img_size, batch_size=bs, label_mode="categorical")
    val=tf.keras.preprocessing.image_dataset_from_directory(
        dirpath, validation_split=vsplit, subset="validation", seed=seed,
        image_size=img_size, batch_size=bs, label_mode="categorical")
    AUTOTUNE=tf.data.AUTOTUNE
    return train.cache().prefetch(AUTOTUNE), val.cache().prefetch(AUTOTUNE), train.class_names

def build_model(key, input_shape=(224,224,3), num_classes=3, lr=1e-3, imagenet_weights=True):
    try: keras.backend.clear_session()
    except Exception: pass
    inputs=keras.Input(shape=input_shape); x=inputs
    if key=="custom_cnn":
        x=layers.Rescaling(1./255)(x)
        for f in [32,64,128]:
            x=layers.Conv2D(f,3,activation="relu")(x); x=layers.MaxPooling2D()(x)
        x=layers.GlobalAveragePooling2D()(x)
    else:
        w="imagenet" if imagenet_weights else None
        try:
            if key=="mobilenetv2":
                base=tf.keras.applications.MobileNetV2(include_top=False,input_shape=input_shape,weights=w)
                prep=tf.keras.applications.mobilenet_v2.preprocess_input
            elif key=="resnet50v2":
                base=tf.keras.applications.ResNet50V2(include_top=False,input_shape=input_shape,weights=w)
                prep=tf.keras.applications.resnet_v2.preprocess_input
            elif key=="xception":
                base=tf.keras.applications.Xception(include_top=False,input_shape=input_shape,weights=w)
                prep=tf.keras.applications.xception.preprocess_input
            elif key=="densenet121":
                base=tf.keras.applications.DenseNet121(include_top=False,input_shape=input_shape,weights=w)
                prep=tf.keras.applications.densenet.preprocess_input
        except Exception:
            # fallback sin pesos
            if key=="mobilenetv2":
                base=tf.keras.applications.MobileNetV2(include_top=False,input_shape=input_shape,weights=None)
                prep=tf.keras.applications.mobilenet_v2.preprocess_input
            elif key=="resnet50v2":
                base=tf.keras.applications.ResNet50V2(include_top=False,input_shape=input_shape,weights=None)
                prep=tf.keras.applications.resnet_v2.preprocess_input
            elif key=="xception":
                base=tf.keras.applications.Xception(include_top=False,input_shape=input_shape,weights=None)
                prep=tf.keras.applications.xception.preprocess_input
            elif key=="densenet121":
                base=tf.keras.applications.DenseNet121(include_top=False,input_shape=input_shape,weights=None)
                prep=tf.keras.applications.densenet.preprocess_input
        x=prep(x); base.trainable=False; x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(0.2)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    model=keras.Model(inputs,outputs,name=key)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def weight_path(key): Path("models").mkdir(exist_ok=True); return Path("models")/f"{key}.weights.h5"
def meta_path(key): Path("models").mkdir(exist_ok=True); return Path("models")/f"{key}.meta.json"
def load_weights_if_exist(model,key):
    wp=weight_path(key);
    if wp.exists(): model.load_weights(str(wp)); return True
    return False
def save_weights(model,key): model.save_weights(str(weight_path(key)))
def save_meta(key, meta:dict):
    with open(meta_path(key), "w") as f: json.dump(meta, f)
def load_meta(key):
    mp=meta_path(key)
    if mp.exists():
        with open(mp,"r") as f: return json.load(f)
    return {}

def evaluate_on_validation(model, val_ds):
    ys_true=[]; ys_prob=[]
    for x,y in val_ds:
        p=model.predict(x,verbose=0)
        ys_true.append(y.numpy()); ys_prob.append(p)
    y_true=np.concatenate(ys_true,axis=0); y_prob=np.concatenate(ys_prob,axis=0)
    return y_true, y_prob

def model_metrics(y_true_1h, y_prob):
    y_true=np.argmax(y_true_1h,axis=1); y_pred=np.argmax(y_prob,axis=1)
    acc=accuracy_score(y_true,y_pred)
    pr, rc, f1, _=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    mcc=matthews_corrcoef(y_true,y_pred) if len(np.unique(y_true))>1 else 0.0
    cm=confusion_matrix(y_true,y_pred)
    return {"accuracy":acc,"precision":pr,"recall":rc,"f1":f1,"mcc":mcc,"cm":cm}

def plot_learning(curves, title):
    fig=plt.figure()
    plt.plot(curves.get("accuracy",[]), label=t("Accuracy"))
    plt.plot(curves.get("val_accuracy",[]), label=t("Accuracy")+" (val)")
    plt.plot(curves.get("loss",[]), label=t("Loss"))
    plt.plot(curves.get("val_loss",[]), label=t("Loss")+" (val)")
    plt.legend(); plt.title(title)
    return fig

def plot_confusion(cm, class_names):
    cmn=cm.astype(float)/cm.sum(axis=1,keepdims=True).clip(min=1)
    fig=plt.figure()
    plt.imshow(cmn, interpolation='nearest'); plt.title(t("Confusion Matrix"))
    plt.colorbar()
    tick=np.arange(len(class_names)); plt.xticks(tick,class_names,rotation=45,ha="right"); plt.yticks(tick,class_names)
    thresh=cmn.max()/2.
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            plt.text(j,i,f"{cmn[i,j]*100:.1f}%",ha="center",va="center",color="white" if cmn[i,j]>thresh else "black")
    plt.ylabel(t("Classes")); plt.xlabel(t("Classes"))
    return fig

def plot_roc(y_true_1h, y_prob, class_names):
    n=len(class_names); fig=plt.figure()
    for i in range(n):
        fpr,tpr,_=roc_curve(y_true_1h[:,i], y_prob[:,i]); roc_auc=auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    fpr_micro,tpr_micro,_=roc_curve(y_true_1h.ravel(), y_prob.ravel()); auc_micro=auc(fpr_micro,tpr_micro)
    plt.plot(fpr_micro,tpr_micro,linestyle="--",label=f"micro (AUC={auc_micro:.2f})")
    plt.plot([0,1],[0,1],color="gray",linestyle=":")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(t("ROC Curves")); plt.legend(loc="lower right")
    return fig

def fig_to_rlimage(fig, width=480):
    buf=io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    img = RLImage(buf); img._restrictSize(width, 9999)
    return img

def dataset_overview(root, val_split, class_names):
    data={}
    for c in class_names:
        count=len(glob.glob(os.path.join(root,c,"*"))); data[c]=count
    total=sum(data.values()) or 1
    df=pd.DataFrame({t("Class"):list(data.keys()),t("Total"):[data[c] for c in data]})
    df[t("% Total")]=df[t("Total")]/total*100
    df[t("Training samples")]= (df[t("Total")]*(1-val_split)).astype(int)
    df[t("Validation samples")]=df[t("Total")]-df[t("Training samples")]
    df[t("% Train")]=df[t("Training samples")]/df[t("Total")].clip(lower=1)*100
    df[t("% Val")]=100-df[t("% Train")]
    return df

def sample_images(root, class_names, wanted=("Healthy","Late","Early")):
    found={}
    for c in class_names:
        lower=c.lower()
        key="Healthy" if ("healthy" in lower or "sana" in lower or "salud" in lower) else ("Late" if "late" in lower or "tard" in lower else ("Early" if "early" in lower or "tempran" in lower else None))
        if key and key not in found:
            imgs=glob.glob(os.path.join(root,c,"*.jpg"))+glob.glob(os.path.join(root,c,"*.png"))
            if imgs: found[key]=imgs[0]
    return [found.get(k) for k in wanted]

# ---------- ENTRENAMIENTO ----------
def run_training(sel, dataset_dir, vsplit, epochs, bs, lr):
    train,val,classes=load_data(dataset_dir,(224,224),vsplit,bs)
    if train is None: return {"status":"no_data"}
    results={"classes":classes,"model_keys":sel,"metrics":{},"curves":{},"val_eval":{},"epochs":{},"source":{}}
    for k in sel:
        m=build_model(k,(224,224,3),len(classes),lr,imagenet_weights=True)
        h=m.fit(train,validation_data=val,epochs=epochs,verbose=1)
        st.session_state.models[k]=m
        hist=h.history
        results["curves"][k]={"accuracy":hist.get("accuracy",[]),"val_accuracy":hist.get("val_accuracy",[]),
                              "loss":hist.get("loss",[]),"val_loss":hist.get("val_loss",[])}
        results["epochs"][k]=int(epochs); results["source"][k]="trained"
        y_true_1h, y_prob = evaluate_on_validation(m,val)
        results["val_eval"][k]={"y_true":y_true_1h,"y_prob":y_prob}
        met=model_metrics(y_true_1h,y_prob); results["metrics"][k]=met
        if st.session_state.get("save_models",False):
            save_weights(m,k)
            save_meta(k,{
                "classes":classes,
                "epochs":int(epochs),
                "curves":results["curves"][k],
                "val_split":vsplit,
                "created_at":datetime.datetime.utcnow().isoformat()+"Z"
            })
    return {"status":"ok","results":results}

def try_load_saved(sel, dataset_dir, vsplit, bs, lr):
    loaded=False
    if "results" not in st.session_state or not st.session_state.results:
        st.session_state.results={"classes":[], "model_keys":[], "metrics":{}, "curves":{}, "val_eval":{}, "epochs":{}, "source":{}}
    res=st.session_state.results

    classes_from_meta=None
    for k in sel:
        mta=load_meta(k)
        if mta.get("classes"): classes_from_meta=mta["classes"]; break
    if classes_from_meta: res["classes"]=classes_from_meta
    else:
        _,_,cls = load_data(dataset_dir,(224,224),vsplit,bs)
        res["classes"]=cls or res.get("classes",[]) or ["Healthy","Early Blight","Late Blight"]

    for k in sel:
        wp=weight_path(k)
        if not wp.exists(): continue
        try:
            m=build_model(k,(224,224,3),len(res["classes"]),lr,imagenet_weights=False)
            if load_weights_if_exist(m,k):
                st.session_state.models[k]=m; loaded=True
                res["model_keys"]=list(set(res.get("model_keys",[])+[k]))
                mta=load_meta(k)
                if mta.get("curves"): res["curves"][k]=mta["curves"]
                if mta.get("epochs"): res["epochs"][k]=int(mta["epochs"])
                res["source"][k]="loaded"
        except Exception as e:
            st.warning(f"⚠️ {model_display(k)}: {e}")

    # Recalcular evaluación si hay val
    train,val,_=load_data(dataset_dir,(224,224),vsplit,bs)
    if val is not None:
        for k,m in st.session_state.models.items():
            try:
                y_true_1h, y_prob = evaluate_on_validation(m,val)
                res["val_eval"][k]={"y_true":y_true_1h,"y_prob":y_prob}
                res["metrics"][k]=model_metrics(y_true_1h,y_prob)
            except Exception as e:
                st.warning(f"{model_display(k)} eval: {e}")
    return loaded

# ---------- INTERPRETACIÓN AUTOMÁTICA ----------
def interpret_curves(curves, met=None, y_true=None, y_prob=None):
    acc=np.array(curves.get("accuracy",[0])); val_acc=np.array(curves.get("val_accuracy",[0]))
    loss=np.array(curves.get("loss",[1])); val_loss=np.array(curves.get("val_loss",[1]))
    notes=[]
    gap=(acc[-1]-val_acc[-1]) if len(acc)==len(val_acc) and len(acc)>0 else 0
    if (len(acc)>0 and len(val_acc)>0 and (gap>0.1)) or (len(loss)>2 and len(val_loss)>2 and (val_loss[-1]-val_loss[-2])>0 and (loss[-1]-loss[-2])<=0):
        notes.append("• "+t("Possible overfitting (large train/val gap and/or rising val loss)."))
    if (len(val_acc)>0 and val_acc[-1]<0.7) or (len(val_loss)>0 and val_loss[-1]>0.8):
        notes.append("• "+t("Possible underfitting (low accuracies and/or high losses)."))
    if (len(val_acc)>0 and val_acc[-1]>=0.85 and abs(gap)<=0.05):
        notes.append("• "+t("Good fit (high accuracies, stable curves, small gap)."))
    if met is not None:
        notes.append(f"• {t('Accuracy')}: {met['accuracy']:.3f} | {t('Precision')}: {met['precision']:.3f} | {t('Recall')}: {met['recall']:.3f} | F1: {met['f1']:.3f} | MCC: {met['mcc']:.3f}")
    if (y_true is not None) and (y_prob is not None):
        try:
            fpr_micro,tpr_micro,_=roc_curve(y_true.ravel(), y_prob.ravel()); auc_micro=auc(fpr_micro,tpr_micro)
            notes.append(f"• AUC micro: {auc_micro:.3f}")
        except Exception:
            pass
    return notes or ["• "+t("Learning Curves: compare train vs val; look for convergence without divergence.")]

# ---------- Canonicalización de clases ----------
def canonical_disease(label: str) -> str:
    n=(label or "").lower()
    n=n.replace("potato___","").replace("__"," ").replace("_"," ").strip()
    if any(w in n for w in ["healthy","sana","salud"]): return "Healthy"
    if any(w in n for w in ["early","tempran"]): return "Early Blight"
    if any(w in n for w in ["late","tard"]): return "Late Blight"
    return clean_heading(label)

# ---------- PDF Helpers ----------
def make_doc(buffer):
    return SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
_styles = getSampleStyleSheet()
def table_df(df, small=False, long=False, colWidths=None, repeatHeader=True):
    data=[list(df.columns)]+df.values.tolist()
    T = LongTable if long else Table
    t=T(data, colWidths=colWidths, repeatRows=1 if repeatHeader else 0)
    ts=[
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#F0F0F0")),
        ("GRID",(0,0),(-1,-1), 0.5, colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUND",(0,1),(-1,-1), colors.white),
        ("WORDWRAP",(0,0),(-1,-1),"CJK"),
    ]
    if small:
        ts.append(("FONTSIZE",(0,0),(-1,-1),8))
        ts.append(("BOTTOMPADDING",(0,0),(-1,-1),4))
        ts.append(("TOPPADDING",(0,0),(-1,-1),4))
    t.setStyle(TableStyle(ts)); return t
def para(text, style="BodyText"): return Paragraph(clean_heading(text), _styles[style])

def pdf_technical(dataset_path, val_split):
    buf=io.BytesIO(); doc=make_doc(buf); elems=[]
    elems.append(Paragraph(clean_heading(t("Technical Report")), _styles["Heading1"]))
    df_specs=get_software_specs_df(st.session_state.lang); elems.append(Paragraph(t("Software Specifications"), _styles["Heading2"]))
    elems.append(table_df(df_specs, colWidths=[160, 320])); elems.append(Spacer(1,10))
    df_libs=get_library_versions_df(st.session_state.lang); elems.append(Paragraph(t("Library Versions"), _styles["Heading2"]))
    elems.append(table_df(df_libs, colWidths=[200, 280])); elems.append(Spacer(1,10))
    class_dirs=[p.name for p in Path(dataset_path).glob("*") if p.is_dir()]
    if class_dirs:
        info=dataset_overview(dataset_path, val_split, class_dirs)
        elems.append(Paragraph(t("Dataset Samples"), _styles["Heading2"]))
        elems.append(table_df(info, small=True, long=True)); elems.append(Spacer(1,10))
    doc.build(elems); return buf.getvalue()

def pdf_training(results, dataset_path, val_split):
    buf=io.BytesIO(); doc=make_doc(buf); elems=[]
    elems.append(Paragraph(clean_heading(t("Training Report")), _styles["Heading1"]))
    classes=results.get("classes",[])
    elems.append(para(f"{t('Classes')}: {', '.join(classes) if classes else '-'}"))
    elems.append(para(f"{t('Dataset path')}: {dataset_path}"))
    elems.append(para(f"{t('Validation split')}: {val_split}")); elems.append(Spacer(1,12))
    rows=[]
    for k in results.get("model_keys",[]):
        met=results["metrics"].get(k);
        if not met: continue
        rows.append([model_display(k), results.get("epochs",{}).get(k,"-"),
                     f"{met['accuracy']:.3f}", f"{met['precision']:.3f}", f"{met['recall']:.3f}", f"{met['f1']:.3f}", f"{met['mcc']:.3f}"])
    if rows:
        df_sum=pd.DataFrame(rows, columns=[t("Models"), t("Epochs used"), t("Accuracy"), t("Precision"), t("Recall"), "F1-Score", "MCC"])
        elems.append(Paragraph(t("Summary"), _styles["Heading2"]))
        elems.append(table_df(df_sum)); elems.append(Spacer(1,12))
    for k in results.get("model_keys",[]):
        if k not in results.get("curves",{}): continue
        elems.append(Paragraph(model_display(k), _styles["Heading2"]))
        source_label = t("Curves source (trained)") if results.get("source",{}).get(k)=="trained" else t("Curves source (loaded)")
        elems.append(para(source_label)); elems.append(Spacer(1,4))
        met=results.get("metrics",{}).get(k); y_true=results.get("val_eval",{}).get(k,{}).get("y_true"); y_prob=results.get("val_eval",{}).get(k,{}).get("y_prob")
        for n in interpret_curves(results["curves"].get(k,{}), met, y_true, y_prob): elems.append(para(n))
        elems.append(Spacer(1,6))
        curves=results["curves"].get(k,{})
        elems.append(para(t("Learning Curves (Accuracy/Loss)"), "Heading3")); elems.append(fig_to_rlimage(plot_learning(curves, t("Learning Curves (Accuracy/Loss)"))))
        if met:
            classes=results["classes"]; elems.append(para(t("Confusion Matrix (Validation)"), "Heading3"))
            elems.append(fig_to_rlimage(plot_confusion(met["cm"], classes)))
            elems.append(para(t("ROC Curves (Validation)"), "Heading3")); elems.append(fig_to_rlimage(plot_roc(y_true,y_prob,classes)))
        elems.append(Spacer(1,10))
    doc.build(elems); return buf.getvalue()

def pdf_graphs_only(results):
    buf=io.BytesIO(); doc=make_doc(buf); elems=[]
    elems.append(Paragraph(clean_heading(t("Graphs Report")), _styles["Heading1"]))
    for k in results.get("model_keys",[]):
        if k not in results.get("curves",{}): continue
        elems.append(Paragraph(model_display(k), _styles["Heading2"]))
        elems.append(para(t("Learning Curves (Accuracy/Loss)"), "Heading3"))
        curves=results["curves"].get(k,{})
        elems.append(fig_to_rlimage(plot_learning(curves, model_display(k))))
        if k in results.get("metrics",{}):
            classes=results["classes"]; y_true=results["val_eval"][k]["y_true"]; y_prob=results["val_eval"][k]["y_prob"]
            elems.append(para(t("Confusion Matrix (Validation)"), "Heading3"))
            elems.append(fig_to_rlimage(plot_confusion(results["metrics"][k]["cm"], classes)))
            elems.append(para(t("ROC Curves (Validation)"), "Heading3"))
            elems.append(fig_to_rlimage(plot_roc(y_true,y_prob,classes)))
        elems.append(Spacer(1,10))
    doc.build(elems); return buf.getvalue()

def pdf_interpretation(results):
    buf=io.BytesIO(); doc=make_doc(buf); elems=[]
    elems.append(Paragraph(clean_heading(t("Interpretation Report")), _styles["Heading1"]))
    elems.append(para("• "+t("ROC: closer to top-left is better; AUC summarizes per-class performance.")))
    elems.append(para("• "+t("Confusion Matrix: high diagonal means good performance; inspect off-diagonal confusions.")))
    elems.append(para("• "+t("Learning Curves: compare train vs val; look for convergence without divergence."))); elems.append(Spacer(1,8))
    for k in results.get("model_keys",[]):
        if k not in results.get("curves",{}): continue
        elems.append(Paragraph(model_display(k), _styles["Heading2"]))
        met=results.get("metrics",{}).get(k); y_true=results.get("val_eval",{}).get(k,{}).get("y_true"); y_prob=results.get("val_eval",{}).get(k,{}).get("y_prob")
        for n in interpret_curves(results["curves"].get(k,{}), met, y_true, y_prob): elems.append(para(n))
        elems.append(Spacer(1,6))
    doc.build(elems); return buf.getvalue()

def pdf_diagnosis_single(model_name, classes, probs, recommendation_lines, image_bytes=None, top_label=None, top_prob=None):
    buf=io.BytesIO(); doc=make_doc(buf); elems=[]
    elems.append(Paragraph(clean_heading(t("Diagnosis Report")), _styles["Heading1"]))
    elems.append(Paragraph(model_name, _styles["Heading2"]))
    if top_label is not None and top_prob is not None:
        elems.append(para(f"{t('Diagnosis result')}: {top_label} — {top_prob*100:.1f}%")); elems.append(Spacer(1,8))
    if image_bytes:
        im = RLImage(io.BytesIO(image_bytes)); im._restrictSize(400, 400); elems.append(im); elems.append(Spacer(1,8))
    df = pd.DataFrame({t("Class"): classes, t("Confidence"): [f"{p*100:.1f}%" for p in probs]})
    elems.append(Paragraph(t("Per-class probabilities"), _styles["Heading3"])); elems.append(table_df(df, colWidths=[260, 160])); elems.append(Spacer(1,8))
    elems.append(Paragraph(t("Recommendation"), _styles["Heading3"]))
    for line in recommendation_lines: elems.append(para("• "+line))
    doc.build(elems); return buf.getvalue()

# ---------- TAB ENTRENAMIENTO ----------
with tabs[1]:
    st.subheader(t("Train"))
    st.caption(f"📁 {t('Dataset path')}: `{dataset_path}`")
    st.write("**"+t("Selected models")+":** "+(", ".join([clean_heading(model_display(k)) for k in selected_models]) if selected_models else "—"))

    # Toggle único: Show / Hide results (sin duplicar "Entrenar modelos")
    toggle_label = f"👁️ {t('Show Results')}" if not st.session_state.show_models else f"🔒 {t('Hide Results')}"
    if st.button(toggle_label, key="toggle_show_models"):
        st.session_state.show_models = not st.session_state.show_models

    # Acciones explícitas
    colA,colB,colC = st.columns(3)
    start = colA.button("🚀 "+t("Train Models"))
    runstats = colB.button("📊 "+t("Run Stats"))
    do_load = colC.button("📦 "+t("Load saved models now")) if load_saved else False

    # Mensajes de verificación (rápidos, sin cálculos)
    if not (start or do_load or st.session_state.show_models):
        if not selected_models and not load_saved:
            st.info("ℹ️ "+t("No models selected and loading not enabled.")+f" {t('Nothing to show yet. Select models or enable loading and press the load button.')}")
        else:
            st.info("ℹ️ "+t("Nothing to show yet. Select models or enable loading and press the load button."))

    # Cargar guardados solo bajo petición y con selección de modelos
    if do_load:
        if not selected_models:
            st.warning("⚠️ "+t("Models")+": 0")
        else:
            with st.spinner("..."):
                ok = try_load_saved(selected_models, dataset_path, val_split, batch_size, learning_rate)
            if ok:
                st.info("ℹ️ "+t("Models loaded. Press 'Show Results' to display results."))

    # Entrenamiento bajo demanda
    if start:
        if not selected_models:
            st.warning("⚠️ "+t("Models")+": 0")
        else:
            with st.spinner("..."):
                out=run_training(selected_models,dataset_path,val_split,epochs,batch_size,learning_rate)
            if out["status"]=="no_data":
                st.error(f"⚠️ {t('No dataset found')}: `{dataset_path}`")
            else:
                st.session_state.results=out["results"]; st.session_state.trained_once=True
                st.success("✅ "+t("Training finished"))
                st.info("ℹ️ "+t("Learning curves note"))

    # Mostrar resultados solo si el usuario lo pide
    res=st.session_state.results
    if st.session_state.show_models and res.get("metrics"):
        for k in selected_models:
            if k not in res["metrics"]: continue
            cur=res["curves"].get(k, {})
            source_label = t("Curves source (trained)") if res.get("source",{}).get(k)=="trained" else t("Curves source (loaded)")
            st.markdown(f"**{t('Learning Curves')} — {clean_heading(model_display(k))} ({t('Epochs used')}: {res.get('epochs',{}).get(k,'-')})**  \n_{source_label}_")
            st.pyplot(plot_learning(cur, t("Learning Curves (Accuracy/Loss)")))
            classes=res["classes"]
            y_true=res["val_eval"][k]["y_true"]; y_prob=res["val_eval"][k]["y_prob"]
            met=res["metrics"][k]
            st.markdown(f"**{model_display(k)}** — {t('Accuracy')}: {met['accuracy']:.3f} | {t('Precision')}: {met['precision']:.3f} | {t('Recall')}: {met['recall']:.3f} | {t('F1-Score')}: {met['f1']:.3f} | {t('MCC')}: {met['mcc']:.3f}")
            st.pyplot(plot_confusion(met["cm"], classes))
            st.pyplot(plot_roc(y_true, y_prob, classes))

    # Estadísticas (tablas limpias y traducidas)
    if runstats:
        st.markdown(f"### {t('Statistics')}")
        usable=[]
        for k in selected_models:
            if res.get("val_eval",{}).get(k) is not None:
                usable.append(k)
        if len(usable)<2:
            st.warning(f"⚠️ {t('Stats need at least 2 models')}. {t('Execution canceled: select more models.')}")
        else:
            by_model=[]; model_names=[]; y_len=None
            for k in usable:
                y1=res["val_eval"][k]["y_true"]; yp=res["val_eval"][k]["y_prob"]
                y=np.argmax(y1,axis=1); yhat=np.argmax(yp,axis=1)
                correct=(yhat==y).astype(int)
                by_model.append(correct); model_names.append(model_display(k))
                y_len=len(correct) if y_len is None else min(y_len,len(correct))
            by_model=[v[:y_len] for v in by_model]
            try:
                stat, p = f_oneway(*by_model)
                df_anova = pd.DataFrame([[float(stat), float(p)]], columns=[t("F-statistic"), t("p-value")], index=[t("ANOVA")])
                st.markdown(f"**{t('ANOVA Summary')}**"); st.dataframe(df_anova, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"ANOVA: {e}")
            rows_mc=[]
            for i in range(len(usable)):
                for j in range(i+1,len(usable)):
                    a=by_model[i]; b=by_model[j]
                    n01=int(np.sum((a==0)&(b==1))); n10=int(np.sum((a==1)&(b==0)))
                    table=[[0,n01],[n10,0]]
                    try:
                        res_mc=mcnemar(table, exact=True); pval=float(res_mc.pvalue)
                    except Exception:
                        pval=np.nan
                    rows_mc.append({t("Model A"):model_names[i],t("Model B"):model_names[j],t("N01"):n01,t("N10"):n10,t("p-value"):pval,t("Significant (α=0.05)"):(pval<0.05) if np.isfinite(pval) else False})
            if rows_mc:
                st.markdown(f"**{t('McNemar (pairs)')}**"); st.dataframe(pd.DataFrame(rows_mc), use_container_width=True, hide_index=True)
            try:
                data=np.concatenate(by_model)
                labels=np.concatenate([[model_names[i]]*y_len for i in range(len(model_names))])
                mc=MultiComparison(data, labels); tuk=mc.tukeyhsd(); tbl=tuk.summary()
                tuk_df=pd.DataFrame(tbl.data[1:], columns=tbl.data[0])
                for col in ["meandiff","p-adj","lower","upper"]: tuk_df[col]=pd.to_numeric(tuk_df[col], errors="coerce")
                tuk_df = tuk_df.rename(columns={"group1": t("Model A"),"group2": t("Model B"),"meandiff": t("Mean Diff"),"p-adj": t("p-value"),"lower": t("Lower CI"),"upper": t("Upper CI"),"reject": t("Reject H0")})
                st.markdown(f"**{t('Tukey HSD')}**"); st.dataframe(tuk_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Tukey HSD: {e}")

# ---------- DIAGNÓSTICO (individual + recomendaciones) ----------
def predict_single(img_pil, model, classes):
    img=img_pil.resize((224,224)).convert("RGB")
    arr=np.expand_dims(np.array(img).astype("float32")/255.0,0)
    prob=model.predict(arr,verbose=0)[0]
    probs = prob / (prob.sum()+1e-8)
    idx=int(np.argmax(probs)); cls_raw=classes[idx] if idx<len(classes) else "N/A"
    cls_can = canonical_disease(cls_raw)
    return cls_raw, cls_can, float(probs[idx]), probs

def recommendation_lines(canonical_disease_name, p, lang="es"):
    base_map={
        "Healthy":{
            "es":["Planta sana: mantener prácticas actuales.","Riego y nutrición equilibrados.","Monitoreo preventivo semanal.","Higiene de herramientas y control de malezas."],
            "en":["Plant appears healthy: keep current practices.","Balanced irrigation and nutrition.","Weekly preventive monitoring.","Tool hygiene and weed control."]
        },
        "Early Blight":{
            "es":["Retirar hojas muy afectadas (desechar).","Mejorar ventilación y reducir humedad foliar.","Evitar riego por aspersión.","Desinfectar herramientas después de la poda."],
            "en":["Remove severely affected leaves (dispose).","Improve airflow and reduce leaf wetness.","Avoid overhead irrigation.","Disinfect tools after pruning."]
        },
        "Late Blight":{
            "es":["Aislar plantas afectadas y monitorear vecinas.","Minimizar humedad en follaje; riego al suelo.","Eliminar restos infectados para cortar diseminación.","Desinfectar herramientas y superficies de trabajo."],
            "en":["Isolate affected plants and monitor neighbors.","Keep foliage dry; irrigate at soil level.","Remove infected debris to reduce spread.","Disinfect tools and work surfaces."]
        }
    }
    mod = {
        "es": {
            "high": {
                "Healthy":["Confianza: Alta. Mantener rotación y espaciamiento.","Registrar fecha/condiciones para trazabilidad."],
                "Early Blight":["Confianza: Alta. Manejo integrado y protectantes permitidos.","Inspección diaria de nuevas lesiones."],
                "Late Blight":["Confianza: Alta. Acción inmediata; limitar dispersión.","Revisión diaria por progresión rápida."]
            },
            "med": {
                "Healthy":["Confianza: Media. Confirmar con nuevas imágenes.","Monitoreo 2–3 veces por semana."],
                "Early Blight":["Confianza: Media. Revalorar en 24–48h; podar sospechosas.","Mejorar ventilación; protectante si progresa."],
                "Late Blight":["Confianza: Media. Aumentar monitoreo y mantener hojas secas.","Considerar protectante preventivo."]
            },
            "low": {
                "Healthy":["Confianza: Baja. Repetir captura bien iluminada.","Evaluar más hojas de la misma planta."],
                "Early Blight":["Confianza: Baja. Buscar lesiones en diana; repetir captura.","Observar 48h antes de intervenir."],
                "Late Blight":["Confianza: Baja. Buscar manchas acuosas/micelio; repetir captura.","Monitoreo cercano hasta confirmar."]
            }
        },
        "en": {
            "high": {
                "Healthy":["Confidence: High. Keep rotation and spacing.","Log date/conditions for traceability."],
                "Early Blight":["Confidence: High. Integrated management with approved protectants.","Daily inspection for new lesions."],
                "Late Blight":["Confidence: High. Immediate action; limit spread.","Daily checks due to fast progression."]
            },
            "med": {
                "Healthy":["Confidence: Medium. Confirm with new images.","Monitor 2–3 times per week."],
                "Early Blight":["Confidence: Medium. Recheck in 24–48h; prune suspicious tissue.","Improve airflow; protectant if it progresses."],
                "Late Blight":["Confidence: Medium. Increase monitoring; keep foliage dry.","Consider preventive protectant."]
            },
            "low": {
                "Healthy":["Confidence: Low. Retake with good lighting.","Evaluate more leaves on the same plant."],
                "Early Blight":["Confidence: Low. Look for target-like lesions; retake.","Observe 48h before intervening."],
                "Late Blight":["Confidence: Low. Check water-soaked spots/mycelium; retake.","Close monitoring until confirmed."]
            }
        }
    }
    lang_code="es" if lang=="es" else "en"
    dz = canonical_disease_name if canonical_disease_name in ["Healthy","Early Blight","Late Blight"] else "Healthy"
    lines=list(base_map[dz][lang_code])
    level="high" if p>=0.9 else ("med" if p>=0.7 else "low")
    lines+=mod[lang_code][level][dz]
    return lines

with tabs[3]:
    st.subheader(t("Diagnosis"))
    res=st.session_state.results
    classes=res.get("classes",[]) or ["Healthy","Early Blight","Late Blight"]

    s_paths=sample_images(dataset_path, classes, ("Healthy","Late","Early")) if classes else [None,None,None]
    cols=st.columns(3)
    for i,(title,pth) in enumerate(zip(["Hoja sana","Tizón tardío (Late blight)","Tizón temprano (Early blight)"], s_paths)):
        if pth and os.path.exists(pth):
            with cols[i]: st.image(pth, caption=title, use_container_width=True)

    loaded_keys=list(st.session_state.models.keys())
    model_choice = st.selectbox(t("Model for diagnosis"), loaded_keys, format_func=lambda k: model_display(k)) if loaded_keys else None

    img=st.file_uploader(t("Upload image"), type=["jpg","jpeg","png"], accept_multiple_files=False)
    if st.button("🩺 "+t("Predict")):
        if not st.session_state.models:
            st.warning(t("No models loaded"))
        elif img is None:
            st.warning("⚠️ "+t("Upload image"))
        elif model_choice is None:
            st.warning("⚠️ "+t("Models"))
        else:
            image=Image.open(img).convert("RGB")
            buf=io.BytesIO(); image.save(buf, format="PNG"); st.session_state.last_img_bytes=buf.getvalue()
            model = st.session_state.models[model_choice]
            cls_raw, cls_can, top_p, probs = predict_single(image, model, classes)

            st.markdown(f"**{t('Diagnosis result')}:** {cls_can} — {top_p*100:.1f}%")
            dfp = pd.DataFrame({t("Class"): classes, t("Confidence"): [f"{pp*100:.1f}%" for pp in probs]})
            st.markdown(f"**{t('Per-class probabilities')} — {model_display(model_choice)}**")
            st.dataframe(dfp, use_container_width=True, hide_index=True)

            lines = recommendation_lines(cls_can, top_p, st.session_state.lang)
            st.subheader(t("Recommendation"))
            for line in lines: st.write(f"• {line}")

            st.session_state.results["last_pred_single"] = {
                "model_name": model_display(model_choice),
                "classes": classes,
                "probs": probs.tolist(),
                "reco_lines": lines,
                "top_label": cls_can,
                "top_prob": top_p
            }

# ---------- REPORTES ----------
with tabs[2]:
    st.subheader(t("Reports"))
    has_training=bool(st.session_state.results.get("metrics"))
    has_pred = st.session_state.results.get("last_pred_single") is not None
    cA,cB,cC,cD,cE = st.columns(5)
    if cA.button("🧾 "+t("Generate Technical PDF")):
        st.session_state.pdfs["tech"]=pdf_technical(dataset_path, val_split); st.success("✅ "+t("Download Technical PDF"))
    if cB.button("🧾 "+t("Generate Training PDF"), disabled=not has_training):
        st.session_state.pdfs["train"]=pdf_training(st.session_state.results, dataset_path, val_split); st.success("✅ "+t("Download Training PDF"))
    if cC.button("🧾 "+t("Generate Graphs PDF"), disabled=not has_training):
        st.session_state.pdfs["graphs"]=pdf_graphs_only(st.session_state.results); st.success("✅ "+t("Download Graphs PDF"))
    if cD.button("🧾 "+t("Generate Interpretation PDF"), disabled=not has_training):
        st.session_state.pdfs["interp"]=pdf_interpretation(st.session_state.results); st.success("✅ "+t("Download Interpretation PDF"))
    if cE.button("🧾 "+t("Generate Diagnosis PDF"), disabled=not has_pred):
        last = st.session_state.results.get("last_pred_single")
        if last:
            st.session_state.pdfs["diag"]=pdf_diagnosis_single(
                last["model_name"], last["classes"], last["probs"], last["reco_lines"],
                image_bytes=st.session_state.last_img_bytes, top_label=last.get("top_label"), top_prob=last.get("top_prob")
            ); st.success("✅ "+t("Download Diagnosis PDF"))
        else:
            st.warning(t("No models loaded"))
    d1,d2,d3,d4,d5=st.columns(5)
    d1.download_button(t("Download Technical PDF"), data=st.session_state.pdfs.get("tech",b""), file_name=f"{clean_heading(t('Technical Report'))}.pdf", mime="application/pdf", disabled="tech" not in st.session_state.pdfs)
    d2.download_button(t("Download Training PDF"), data=st.session_state.pdfs.get("train",b""), file_name=f"{clean_heading(t('Training Report'))}.pdf", mime="application/pdf", disabled="train" not in st.session_state.pdfs)
    d3.download_button(t("Download Diagnosis PDF"), data=st.session_state.pdfs.get("diag",b""), file_name=f"{clean_heading(t('Diagnosis Report'))}.pdf", mime="application/pdf", disabled="diag" not in st.session_state.pdfs)
    d4.download_button(t("Download Graphs PDF"), data=st.session_state.pdfs.get("graphs",b""), file_name=f"{clean_heading(t('Graphs Report'))}.pdf", mime="application/pdf", disabled="graphs" not in st.session_state.pdfs)
    d5.download_button(t("Download Interpretation PDF"), data=st.session_state.pdfs.get("interp",b""), file_name=f"{clean_heading(t('Interpretation Report'))}.pdf", mime="application/pdf", disabled="interp" not in st.session_state.pdfs)

# ---------- MANUAL (100% bilingüe) ----------
with tabs[4]:
    st.subheader(t("User Manual"))
    st.markdown(f"""
**{t('A. Parameters')}**
- **{t('Dataset path:')}** `{dataset_path}`
- {t('Validation split / Epochs / Batch size / Learning rate.')}

**{t('B. Load/Save')}**
- {t('Load Saved Models: check the box, then press 📦 Load saved models now.')}
- {t('Save Trained Models: save weights and metadata after training.')}
- {t('Learning curves note (loaded models may restore curves/epochs from metadata).')}

**{t('C. Recommended flow')}**
1) {t('Select Models.')}
2) {t('Press 🚀 Train Models (or enable loading and press 📦).')}
3) {t('Then press 👁️ Show Results to see curves, CM, ROC and metrics.')}
4) {t('Use 📊 Run Stats to compare.')}
5) {t('Generate PDFs from Reports.')}

**{t('D. Diagnosis')}**
- {t('Single-model diagnosis.')}
- {t('You will see Diagnosis result, per-class table and recommendations tailored by disease and confidence.')}
- {t('The PDF includes the analyzed image.')}
""")
