"""
main.py - API de Perfectime

Carga el modelo XGBoost ya entrenado (ver notebooks/Sumativa.ipynb) y lo expone
como una API para que el frontend pueda pedirle predicciones.

Para correrla:
    uvicorn main:app --reload --port 8000

Documentación interactiva (la genera FastAPI solo):
    http://127.0.0.1:8000/docs
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from imblearn.over_sampling import SMOTE
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Cargar el modelo y los archivos que genera el notebook ---

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "modelos"
DATASETS_DIR = BASE_DIR.parent / "datasets"
DATASET_ORIGINAL = DATASETS_DIR / "ai4i2020.csv"
DATASET_NUEVOS = DATASETS_DIR / "nuevos_registros.csv"

# Este es el mismo orden de columnas que se usó al entrenar el modelo.
COLUMNAS_DEFAULT = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_L",
    "Type_M",
]

UMBRAL_DECISION = 0.50  # a partir de aca se considera "Falla"

try:
    modelo = joblib.load(ARTIFACTS_DIR / "modelo_xgboost.pkl")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"No se encontró modelo_xgboost.pkl en {ARTIFACTS_DIR}. "
        "Hay que correr primero el notebook completo para generarlo."
    ) from exc

# El scaler es necesario porque el modelo se entrenó con los datos escalados.
scaler_path = ARTIFACTS_DIR / "scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None
if scaler is None:
    print("[ADVERTENCIA] no se encontró scaler.pkl, las predicciones van a salir mal.")

columnas_path = ARTIFACTS_DIR / "columnas.pkl"
COLUMNAS = joblib.load(columnas_path) if columnas_path.exists() else COLUMNAS_DEFAULT

# SHAP es opcional: si no está instalado, la API igual funciona, solo que
# la predicción viene sin el desglose de "por qué".
try:
    import shap
    explainer = shap.TreeExplainer(modelo)
except Exception as exc:
    explainer = None
    print(f"[INFO] SHAP no disponible ({exc}), no se va a mostrar el desglose de factores.")

# Nombres en español para mostrar en el frontend
ETIQUETAS_VARIABLES = {
    "Air_temperature_K": "Temperatura del aire",
    "Process_temperature_K": "Temperatura del proceso",
    "Rotational_speed_rpm": "Velocidad de rotación",
    "Torque_Nm": "Torque",
    "Tool_wear_min": "Desgaste de herramienta",
    "Type_L": "Tipo de producto (L)",
    "Type_M": "Tipo de producto (M)",
}

# --- Historial de predicciones (SQLite) ---
# Cada vez que se hace una predicción (individual o por lote) se guarda una
# fila acá. Sirve para las pantallas de Historial y Monitoreo del frontend.

DB_PATH = BASE_DIR / "historial.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            origen TEXT NOT NULL,
            tipo_producto TEXT,
            air_temperature_k REAL,
            process_temperature_k REAL,
            rotational_speed_rpm REAL,
            torque_nm REAL,
            tool_wear_min REAL,
            probabilidad_porcentaje REAL,
            prediccion INTEGER,
            etiqueta TEXT,
            nivel_riesgo TEXT,
            factor_principal TEXT,
            factores_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reentrenamientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            registros_nuevos INTEGER,
            total_registros_entrenamiento INTEGER,
            accuracy REAL,
            precision_ REAL,
            recall REAL,
            f1 REAL,
            auc_roc REAL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def guardar_prediccion(origen, datos, probabilidad_porcentaje, prediccion, etiqueta,
                        nivel_riesgo, factor_principal, factores):
    conn = get_conn()
    cursor = conn.execute(
        """INSERT INTO predicciones
           (fecha, origen, tipo_producto, air_temperature_k, process_temperature_k,
            rotational_speed_rpm, torque_nm, tool_wear_min, probabilidad_porcentaje,
            prediccion, etiqueta, nivel_riesgo, factor_principal, factores_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            origen,
            datos.type,
            datos.air_temperature_k,
            datos.process_temperature_k,
            datos.rotational_speed_rpm,
            datos.torque_nm,
            datos.tool_wear_min,
            probabilidad_porcentaje,
            prediccion,
            etiqueta,
            nivel_riesgo,
            factor_principal,
            json.dumps([f.model_dump() for f in factores]),
        ),
    )
    conn.commit()
    nuevo_id = cursor.lastrowid
    conn.close()
    return nuevo_id


def leer_historial(limite=200):
    conn = get_conn()
    filas = conn.execute(
        "SELECT * FROM predicciones ORDER BY id DESC LIMIT ?", (limite,)
    ).fetchall()
    conn.close()
    resultado = []
    for fila in filas:
        item = dict(fila)
        item["factores"] = json.loads(item.pop("factores_json") or "[]")
        resultado.append(item)
    return resultado


def guardar_reentrenamiento(registros_nuevos, total_registros, metricas):
    conn = get_conn()
    conn.execute(
        """INSERT INTO reentrenamientos
           (fecha, registros_nuevos, total_registros_entrenamiento, accuracy, precision_, recall, f1, auc_roc)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            registros_nuevos,
            total_registros,
            metricas["accuracy"],
            metricas["precision"],
            metricas["recall"],
            metricas["f1"],
            metricas["auc_roc"],
        ),
    )
    conn.commit()
    conn.close()


def leer_reentrenamientos(limite=50):
    conn = get_conn()
    filas = conn.execute(
        "SELECT * FROM reentrenamientos ORDER BY id DESC LIMIT ?", (limite,)
    ).fetchall()
    conn.close()
    return [dict(f) for f in filas]


# --- App de FastAPI ---

app = FastAPI(
    title="Perfectime - API de Mantenimiento Predictivo",
    description="Recibe datos de sensores y devuelve la probabilidad de falla de la máquina.",
    version="1.0.0",
)

# CORS abierto porque esto corre local, para el curso. En un caso real
# habría que restringirlo solo al dominio del frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Qué datos recibe y qué datos devuelve la API ---

class SensorInput(BaseModel):
    # Los rangos (ge/le) son los valores mínimo y máximo que vimos en el
    # notebook al hacer el describe() del dataset, así no dejamos pasar
    # datos que no tienen sentido físico.
    air_temperature_k: float = Field(..., ge=250, le=350)
    process_temperature_k: float = Field(..., ge=250, le=360)
    rotational_speed_rpm: float = Field(..., ge=0, le=5000)
    torque_nm: float = Field(..., ge=0, le=200)
    tool_wear_min: float = Field(..., ge=0, le=500)
    type: Literal["H", "L", "M"] = Field("M")


class Factor(BaseModel):
    variable: str
    etiqueta: str
    valor: float
    contribucion: float
    efecto: Literal["aumenta", "reduce"]


class PredictionOutput(BaseModel):
    id: Optional[int] = None
    prediccion: int
    etiqueta: str
    probabilidad_falla: float
    probabilidad_porcentaje: float
    nivel_riesgo: str
    umbral: float
    factores: List[Factor] = []
    factor_principal: Optional[str] = None


# Para reentrenar hace falta el resultado REAL (si la máquina falló o no),
# no solo los datos de sensores — por eso este esquema es distinto a
# SensorInput: sin el resultado real no se puede "enseñar" nada al modelo.
class RegistroHistorico(SensorInput):
    fallo_real: Literal[0, 1]


# --- Funciones que usan los endpoints ---

def construir_dataframe(datos: SensorInput) -> pd.DataFrame:
    # Hacemos a mano el mismo preprocesamiento del notebook (One-Hot +
    # mismo orden de columnas), porque el modelo espera exactamente eso.
    fila = {
        "Air_temperature_K": datos.air_temperature_k,
        "Process_temperature_K": datos.process_temperature_k,
        "Rotational_speed_rpm": datos.rotational_speed_rpm,
        "Torque_Nm": datos.torque_nm,
        "Tool_wear_min": datos.tool_wear_min,
        "Type_L": 1 if datos.type == "L" else 0,
        "Type_M": 1 if datos.type == "M" else 0,
    }
    df = pd.DataFrame([fila])

    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = 0
    return df[COLUMNAS]


def clasificar_riesgo(prob: float) -> str:
    if prob >= 0.67:
        return "Falla"
    if prob >= 0.34:
        return "Riesgo medio"
    return "No falla"


def calcular_factores_lote(df_escalado: pd.DataFrame, df_original: pd.DataFrame) -> List[List[Factor]]:
    # Mismo SHAP que en el notebook (sección 9), pero sirve para 1 o varios registros a la vez.
    n = len(df_original)
    if explainer is None:
        return [[] for _ in range(n)]
    try:
        valores = np.asarray(explainer.shap_values(df_escalado))
        if valores.ndim == 1:
            valores = valores.reshape(1, -1)

        salida = []
        for i in range(n):
            factores = [
                Factor(
                    variable=col,
                    etiqueta=ETIQUETAS_VARIABLES.get(col, col),
                    valor=float(df_original.iloc[i][col]),
                    contribucion=round(float(valores[i][j]), 4),
                    efecto="aumenta" if valores[i][j] >= 0 else "reduce",
                )
                for j, col in enumerate(COLUMNAS)
            ]
            salida.append(sorted(factores, key=lambda f: abs(f.contribucion), reverse=True))
        return salida
    except Exception as exc:
        print(f"[INFO] no se pudieron calcular los valores SHAP: {exc}")
        return [[] for _ in range(n)]


def calcular_factores(df_escalado: pd.DataFrame, df_original: pd.DataFrame) -> List[Factor]:
    return calcular_factores_lote(df_escalado, df_original)[0]


# --- Reentrenamiento del modelo con datos históricos nuevos ---
# Mismo pipeline que el notebook (sección 5 y 6): limpiar columnas, One-Hot,
# split 70/30, escalar y balancear con SMOTE antes de entrenar.

def cargar_dataset_combinado() -> pd.DataFrame:
    df = pd.read_csv(DATASET_ORIGINAL)
    df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
    df.columns = [c.replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]

    if DATASET_NUEVOS.exists():
        df_nuevos = pd.read_csv(DATASET_NUEVOS)
        df = pd.concat([df, df_nuevos], ignore_index=True)

    return df


def preparar_xy(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    for col in ["Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0
    X = df[COLUMNAS_DEFAULT]
    y = df["Machine_failure"]
    return X, y


def reentrenar_modelo(registros_nuevos: List["RegistroHistorico"]):
    # 1. Guardar los registros nuevos en su propio CSV (se van acumulando;
    #    el dataset original ai4i2020.csv nunca se toca).
    filas_nuevas = pd.DataFrame([{
        "Type": r.type,
        "Air_temperature_K": r.air_temperature_k,
        "Process_temperature_K": r.process_temperature_k,
        "Rotational_speed_rpm": r.rotational_speed_rpm,
        "Torque_Nm": r.torque_nm,
        "Tool_wear_min": r.tool_wear_min,
        "Machine_failure": r.fallo_real,
    } for r in registros_nuevos])

    if DATASET_NUEVOS.exists():
        filas_nuevas = pd.concat([pd.read_csv(DATASET_NUEVOS), filas_nuevas], ignore_index=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    filas_nuevas.to_csv(DATASET_NUEVOS, index=False)

    # 2. Armar el dataset combinado (original + todos los nuevos hasta ahora)
    df_completo = cargar_dataset_combinado()
    X, y = preparar_xy(df_completo)

    # 3. Mismo split y balanceo que en el notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    nuevo_scaler = StandardScaler()
    X_train_scaled = nuevo_scaler.fit_transform(X_train)
    X_test_scaled = nuevo_scaler.transform(X_test)
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

    # 4. Reentrenar con la misma configuración que usa el modelo en producción
    #    (ver notebook, sección 7.2 y 10 — el modelo servido es el XGBoost base,
    #    sin los hiperparámetros de GridSearchCV).
    nuevo_modelo = XGBClassifier(random_state=42, eval_metric="logloss")
    nuevo_modelo.fit(X_train_bal, y_train_bal)

    y_pred = nuevo_modelo.predict(X_test_scaled)
    y_proba = nuevo_modelo.predict_proba(X_test_scaled)[:, 1]
    metricas = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    # 5. Reemplazar los artefactos guardados en disco...
    joblib.dump(nuevo_modelo, ARTIFACTS_DIR / "modelo_xgboost.pkl")
    joblib.dump(nuevo_scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(COLUMNAS_DEFAULT, ARTIFACTS_DIR / "columnas.pkl")

    # ...y también los que la API ya tiene cargados en memoria, para que
    # /predict use el modelo nuevo inmediatamente, sin reiniciar el servidor.
    global modelo, scaler, COLUMNAS, explainer
    modelo = nuevo_modelo
    scaler = nuevo_scaler
    COLUMNAS = COLUMNAS_DEFAULT
    try:
        explainer = shap.TreeExplainer(modelo)
    except Exception:
        explainer = None

    guardar_reentrenamiento(len(registros_nuevos), len(df_completo), metricas)

    return {
        "registros_nuevos": len(registros_nuevos),
        "total_registros_entrenamiento": len(df_completo),
        "metricas_modelo_nuevo": metricas,
    }


# --- Endpoints ---

@app.get("/", tags=["Estado"])
def raiz():
    return {
        "servicio": "Perfectime - API de Mantenimiento Predictivo",
        "modelo": type(modelo).__name__,
        "variables_esperadas": COLUMNAS,
        "documentacion": "/docs",
    }


@app.get("/health", tags=["Estado"])
def health():
    return {
        "estado": "ok",
        "modelo_cargado": modelo is not None,
        "scaler_cargado": scaler is not None,
        "shap_disponible": explainer is not None,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predicción"])
def predecir(datos: SensorInput):
    # Predicción para una sola máquina (la usa la pantalla "Predicción individual")
    try:
        df = construir_dataframe(datos)
        df_escalado = (
            pd.DataFrame(scaler.transform(df), columns=COLUMNAS)
            if scaler is not None else df
        )

        probabilidad = float(modelo.predict_proba(df_escalado)[0, 1])
        prediccion = int(probabilidad >= UMBRAL_DECISION)
        factores = calcular_factores(df_escalado, df)
        factor_principal = factores[0].etiqueta if factores else None

        resultado = PredictionOutput(
            prediccion=prediccion,
            etiqueta="Falla" if prediccion == 1 else "No falla",
            probabilidad_falla=round(probabilidad, 4),
            probabilidad_porcentaje=round(probabilidad * 100, 2),
            nivel_riesgo=clasificar_riesgo(probabilidad),
            umbral=UMBRAL_DECISION,
            factores=factores,
            factor_principal=factor_principal,
        )

        resultado.id = guardar_prediccion(
            "individual", datos, resultado.probabilidad_porcentaje,
            resultado.prediccion, resultado.etiqueta, resultado.nivel_riesgo,
            factor_principal, factores)

        return resultado

    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Error al generar la predicción: {exc}") from exc


@app.post("/predict-batch", tags=["Predicción"])
def predecir_lote(registros: List[SensorInput]):
    # Predicción para varios registros de una vez (para cuando se sube un CSV)
    if not registros:
        raise HTTPException(status_code=400, detail="La lista de registros está vacía.")
    if len(registros) > 5000:
        raise HTTPException(status_code=413, detail="Máximo 5.000 registros por solicitud.")

    df = pd.concat([construir_dataframe(r) for r in registros], ignore_index=True)
    df_escalado = (
        pd.DataFrame(scaler.transform(df), columns=COLUMNAS)
        if scaler is not None else df
    )
    probabilidades = modelo.predict_proba(df_escalado)[:, 1]
    factores_por_fila = calcular_factores_lote(df_escalado, df)

    resultados = []
    for i, p in enumerate(probabilidades):
        prediccion = int(p >= UMBRAL_DECISION)
        etiqueta = "Falla" if prediccion == 1 else "No falla"
        nivel_riesgo = clasificar_riesgo(float(p))
        factores = factores_por_fila[i]
        factor_principal = factores[0].etiqueta if factores else None

        resultados.append({
            "indice": i,
            "probabilidad_falla": round(float(p), 4),
            "probabilidad_porcentaje": round(float(p) * 100, 2),
            "prediccion": prediccion,
            "etiqueta": etiqueta,
            "nivel_riesgo": nivel_riesgo,
            "factor_principal": factor_principal,
        })

        guardar_prediccion("lote", registros[i], round(float(p) * 100, 2), prediccion,
                           etiqueta, nivel_riesgo, factor_principal, factores)

    return {
        "total_registros": len(resultados),
        "fallas_detectadas": sum(r["prediccion"] for r in resultados),
        "probabilidad_promedio": round(float(probabilidades.mean() * 100), 2),
        "resultados": resultados,
    }


@app.get("/historial", tags=["Historial"])
def historial(limite: int = 200):
    return {"registros": leer_historial(limite)}


@app.post("/reentrenar", tags=["Reentrenamiento"])
def reentrenar(registros: List[RegistroHistorico]):
    # Reentrena el modelo sumando datos históricos NUEVOS con resultado real
    # conocido (fallo_real). Sin ese resultado real no hay forma correcta de
    # "enseñarle" nada al modelo — por eso este endpoint pide más que solo
    # los datos de sensores que pide /predict.
    if len(registros) < 10:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 10 registros nuevos (con fallo_real) para reentrenar el modelo."
        )
    if len(registros) > 5000:
        raise HTTPException(status_code=413, detail="Máximo 5.000 registros por solicitud.")

    try:
        return reentrenar_modelo(registros)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar el modelo: {exc}") from exc


@app.get("/reentrenamientos", tags=["Reentrenamiento"])
def reentrenamientos(limite: int = 50):
    return {"registros": leer_reentrenamientos(limite)}


@app.get("/monitoreo", tags=["Monitoreo"])
def monitoreo():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) AS n FROM predicciones").fetchone()["n"]
    fallas = conn.execute(
        "SELECT COUNT(*) AS n FROM predicciones WHERE prediccion = 1"
    ).fetchone()["n"]
    promedio = conn.execute(
        "SELECT AVG(probabilidad_porcentaje) AS p FROM predicciones"
    ).fetchone()["p"]
    por_origen = conn.execute(
        "SELECT origen, COUNT(*) AS n FROM predicciones GROUP BY origen"
    ).fetchall()
    por_riesgo = conn.execute(
        "SELECT nivel_riesgo, COUNT(*) AS n FROM predicciones GROUP BY nivel_riesgo"
    ).fetchall()
    ultimo_reentrenamiento = conn.execute(
        "SELECT * FROM reentrenamientos ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    return {
        "total_predicciones": total,
        "fallas_detectadas": fallas,
        "probabilidad_promedio": round(promedio, 2) if promedio is not None else 0,
        "predicciones_por_origen": {r["origen"]: r["n"] for r in por_origen},
        "predicciones_por_riesgo": {r["nivel_riesgo"]: r["n"] for r in por_riesgo},
        "modelo_cargado": modelo is not None,
        "scaler_cargado": scaler is not None,
        "shap_disponible": explainer is not None,
        "ultimo_reentrenamiento": dict(ultimo_reentrenamiento) if ultimo_reentrenamiento else None,
    }
