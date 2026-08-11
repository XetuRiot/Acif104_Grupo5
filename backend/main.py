"""
main.py — API REST de Perfectime
Servicio de inferencia para el modelo XGBoost de mantenimiento predictivo
entrenado sobre el AI4I 2020 Predictive Maintenance Dataset.

Ejecución:
    uvicorn main:app --reload --port 8000
Documentación interactiva:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================================================
# 1. Configuración y carga de artefactos
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

# Orden de columnas por defecto (debe coincidir con el del entrenamiento)
COLUMNAS_DEFAULT = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_L",
    "Type_M",
]

UMBRAL_DECISION = 0.50   # umbral binario Falla / No falla

try:
    modelo = joblib.load(ARTIFACTS_DIR / "modelo_xgboost.pkl")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"No se encontró modelo_xgboost.pkl en {ARTIFACTS_DIR}. "
        "Ejecuta primero la celda de serialización del notebook."
    ) from exc

# El escalador es obligatorio: el modelo fue entrenado con datos estandarizados.
scaler_path = ARTIFACTS_DIR / "scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None
if scaler is None:
    print("[ADVERTENCIA] scaler.pkl no encontrado. Las predicciones serán incorrectas "
          "si el modelo fue entrenado sobre datos escalados.")

columnas_path = ARTIFACTS_DIR / "columnas.pkl"
COLUMNAS = joblib.load(columnas_path) if columnas_path.exists() else COLUMNAS_DEFAULT

# Explicador SHAP (opcional). Si la librería no está disponible, la API sigue
# funcionando y simplemente no devuelve el desglose de factores.
try:
    import shap
    explainer = shap.TreeExplainer(modelo)
except Exception as exc:  # noqa: BLE001
    explainer = None
    print(f"[INFO] SHAP no disponible ({exc}). Se omitirá el desglose de factores.")

# Nombres legibles para la interfaz
ETIQUETAS_VARIABLES = {
    "Air_temperature_K": "Temperatura del aire",
    "Process_temperature_K": "Temperatura del proceso",
    "Rotational_speed_rpm": "Velocidad de rotación",
    "Torque_Nm": "Torque",
    "Tool_wear_min": "Desgaste de herramienta",
    "Type_L": "Tipo de producto (L)",
    "Type_M": "Tipo de producto (M)",
}

# =========================================================
# 2. Aplicación FastAPI
# =========================================================

app = FastAPI(
    title="Perfectime — API de Mantenimiento Predictivo",
    description="Estima la probabilidad de falla de un equipo industrial "
                "a partir de sus variables operacionales.",
    version="1.0.0",
)

# CORS abierto: entorno de desarrollo académico.
# En producción se debe restringir a los dominios del frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 3. Esquemas de entrada y salida
# =========================================================

class SensorInput(BaseModel):
    """Lectura de sensores de una máquina en un instante determinado."""

    air_temperature_k: float = Field(
        ..., ge=250, le=350, description="Temperatura del aire en Kelvin",
        json_schema_extra={"example": 298.1})
    process_temperature_k: float = Field(
        ..., ge=250, le=360, description="Temperatura del proceso en Kelvin",
        json_schema_extra={"example": 308.6})
    rotational_speed_rpm: float = Field(
        ..., ge=0, le=5000, description="Velocidad de rotación en rpm",
        json_schema_extra={"example": 1551})
    torque_nm: float = Field(
        ..., ge=0, le=200, description="Torque en Newton-metro",
        json_schema_extra={"example": 42.8})
    tool_wear_min: float = Field(
        ..., ge=0, le=500, description="Desgaste acumulado de la herramienta en minutos",
        json_schema_extra={"example": 108})
    type: Literal["H", "L", "M"] = Field(
        "M", description="Calidad del producto: H (alta), L (baja) o M (media)")


class Factor(BaseModel):
    variable: str
    etiqueta: str
    valor: float
    contribucion: float
    efecto: Literal["aumenta", "reduce"]


class PredictionOutput(BaseModel):
    prediccion: int
    etiqueta: str
    probabilidad_falla: float
    probabilidad_porcentaje: float
    nivel_riesgo: str
    umbral: float
    factores: List[Factor] = []
    factor_principal: Optional[str] = None


# =========================================================
# 4. Funciones auxiliares
# =========================================================

def construir_dataframe(datos: SensorInput) -> pd.DataFrame:
    """
    Replica exactamente el preprocesamiento del notebook:
      1. Renombra las variables al formato normalizado (snake_case sin corchetes).
      2. Aplica One-Hot Encoding con drop_first=True (categoría H de referencia).
      3. Reordena las columnas según el orden usado en el entrenamiento.
    """
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

    # Garantiza que existan todas las columnas y en el orden correcto
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


def calcular_factores(df_escalado: pd.DataFrame, df_original: pd.DataFrame) -> List[Factor]:
    """Devuelve la contribución SHAP de cada variable, ordenada por magnitud."""
    if explainer is None:
        return []
    try:
        valores = explainer.shap_values(df_escalado)
        valores = np.asarray(valores).reshape(-1)
        factores = [
            Factor(
                variable=col,
                etiqueta=ETIQUETAS_VARIABLES.get(col, col),
                valor=float(df_original.iloc[0][col]),
                contribucion=round(float(v), 4),
                efecto="aumenta" if v >= 0 else "reduce",
            )
            for col, v in zip(COLUMNAS, valores)
        ]
        return sorted(factores, key=lambda f: abs(f.contribucion), reverse=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] No se pudieron calcular los valores SHAP: {exc}")
        return []


# =========================================================
# 5. Endpoints
# =========================================================

@app.get("/", tags=["Estado"])
def raiz():
    return {
        "servicio": "Perfectime — API de Mantenimiento Predictivo",
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
    """Devuelve la probabilidad de falla y la clasificación de riesgo."""
    try:
        df = construir_dataframe(datos)
        df_escalado = (
            pd.DataFrame(scaler.transform(df), columns=COLUMNAS)
            if scaler is not None else df
        )

        probabilidad = float(modelo.predict_proba(df_escalado)[0, 1])
        prediccion = int(probabilidad >= UMBRAL_DECISION)
        factores = calcular_factores(df_escalado, df)

        return PredictionOutput(
            prediccion=prediccion,
            etiqueta="Falla" if prediccion == 1 else "No falla",
            probabilidad_falla=round(probabilidad, 4),
            probabilidad_porcentaje=round(probabilidad * 100, 2),
            nivel_riesgo=clasificar_riesgo(probabilidad),
            umbral=UMBRAL_DECISION,
            factores=factores,
            factor_principal=factores[0].etiqueta if factores else None,
        )

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"Error al generar la predicción: {exc}") from exc


@app.post("/predict-batch", tags=["Predicción"])
def predecir_lote(registros: List[SensorInput]):
    """Predicción para múltiples registros (usado por la carga de archivo CSV)."""
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

    resultados = [
        {
            "indice": i,
            "probabilidad_falla": round(float(p), 4),
            "probabilidad_porcentaje": round(float(p) * 100, 2),
            "prediccion": int(p >= UMBRAL_DECISION),
            "etiqueta": "Falla" if p >= UMBRAL_DECISION else "No falla",
            "nivel_riesgo": clasificar_riesgo(float(p)),
        }
        for i, p in enumerate(probabilidades)
    ]

    return {
        "total_registros": len(resultados),
        "fallas_detectadas": sum(r["prediccion"] for r in resultados),
        "probabilidad_promedio": round(float(probabilidades.mean() * 100), 2),
        "resultados": resultados,
    }