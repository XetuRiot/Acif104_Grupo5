# Perfectime — Mantenimiento Predictivo de Equipos Industriales

Proyecto para la asignatura **Aprendizaje de Máquina (ACIF104)** — Universidad Andrés Bello.

Plataforma de mantenimiento predictivo que estima la probabilidad de falla de un equipo
industrial a partir de variables de sensores, usando un modelo XGBoost servido por una
API REST (FastAPI) y consumido desde una interfaz web.

**Repositorio:** https://github.com/XetuRiot/Acif104_Grupo5

## Integrantes

Javiera Chávez · Joaquín Olivares · Claudio Yáñez

## Estructura

```
Acif104_Grupo5/
├── datasets/ai4i2020.csv         # Dataset (10.000 registros)
├── notebooks/Sumativa.ipynb      # EDA, entrenamiento, evaluación y SHAP
├── modelos/                      # Modelo + scaler + columnas (se generan al ejecutar el notebook)
├── backend/main.py               # API REST (FastAPI)
├── frontend/index_1.html         # Interfaz de operador
├── Documentacion/                # Informes de cada fase
└── environment.yml               # Entorno Conda
```

## Cómo ejecutarlo

### 1. Entorno

```bash
conda env create -f environment.yml
conda activate ml
pip install fastapi "uvicorn[standard]" joblib
```

> Si falla por la línea `prefix:` del `environment.yml`, bórrala y vuelve a intentar —Conda usará la ruta por defecto del equipo.

### 2. Entrenar el modelo

Abrir `notebooks/Sumativa.ipynb` con el kernel del entorno `ml` y ejecutar todas las celdas (**Run All**). Al terminar quedan generados `modelo_xgboost.pkl`, `scaler.pkl` y `columnas.pkl` en `modelos/`.

### 3. Levantar la API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Documentación interactiva en `http://127.0.0.1:8000/docs`, estado del servicio en `http://127.0.0.1:8000/health`.

### 4. Abrir el frontend

Con la API corriendo, abrir `frontend/index_1.html` (recomendado con la extensión **Live Server** de VS Code para evitar restricciones de `file://`). La pantalla de "Predicción individual" consume la API real; el resto de pantallas aún usa datos simulados.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10.000 registros, 96,61 % sin falla / 3,39 % con falla.

Variables predictoras: temperatura del aire, temperatura del proceso, velocidad de rotación, torque, desgaste de herramienta y calidad del producto (Type). Se eliminan `UDI`, `Product ID` (identificadores) y `TWF/HDF/PWF/OSF/RNF` (indican el tipo de falla ocurrida — dejarlas produce fuga de datos).

## Modelos y técnicas

- **Machine Learning:** Random Forest, XGBoost (modelo de producción), Regresión Logística
- **Deep Learning:** 3 arquitecturas de Perceptrón Multicapa (superficial, profunda, ancha)
- **Balanceo de clases:** sin balanceo, submuestreo, SMOTE (comparadas)
- **Ajuste de hiperparámetros:** GridSearchCV sobre XGBoost
- **Interpretabilidad:** SHAP (TreeExplainer)

## Tecnologías

Python 3.12 · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · FastAPI · Uvicorn · JavaScript

## Documentación adicional

- [GUIA_TECNICA.md](GUIA_TECNICA.md) — explicación pedagógica de cada parte del proyecto
- [Documentacion/](Documentacion/) — informes entregados en cada fase del curso

## Licencia

Proyecto académico. Dataset de uso público (UCI Machine Learning Repository).
