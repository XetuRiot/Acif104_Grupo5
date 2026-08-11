# Perfectime — Mantenimiento Predictivo de Equipos Industriales

Proyecto desarrollado para la asignatura **Aprendizaje de Máquina (ACIF104)** — Universidad Andrés Bello.

**Repositorio:** [https://github.com/XetuRiot/Acif104_Grupo5]

## Integrantes

- Javiera Chávez
- Joaquín Olivares
- Claudio Yáñez

## Descripción

Perfectime es una plataforma de mantenimiento predictivo que estima la probabilidad de falla
de equipos industriales a partir de variables operacionales de sensores. El proyecto integra
un modelo supervisado (XGBoost) entrenado sobre el *AI4I 2020 Predictive Maintenance Dataset*,
una API REST desarrollada en FastAPI y una interfaz web de operador.

## Estructura del repositorio

```
Acif104_Grupo5/
├── data/
│   └── ai4i2020.csv              # Dataset (10.000 registros, 14 columnas)
├── notebooks/
│   └── Sumativa.ipynb            # EDA, preprocesamiento, entrenamiento y SHAP
├── backend/
│   ├── main.py                   # API REST (FastAPI)
│   ├── modelo_xgboost.pkl        # Modelo entrenado (generado por el notebook)
│   ├── scaler.pkl                # StandardScaler ajustado
│   └── columnas.pkl              # Orden de columnas esperado por el modelo
├── frontend/
│   └── index_1.html              # Interfaz de operador
├── environment.yml               # Definición del entorno Conda
└── README.md
```

> **Nota:** los tres archivos `.pkl` se generan al ejecutar el notebook (sección 10) y se
> guardan directamente en `backend/`, junto a `main.py`. No se versionan aparte porque
> se regeneran cada vez que se reentrena el modelo.

## Requisitos previos

- **Miniconda** o **Anaconda** ([descarga](https://www.anaconda.com/download))
- **Git**
- Python 3.12 (lo instala el propio entorno)

## 1. Clonar el repositorio

```bash
git clone [https://github.com/XetuRiot/Acif104_Grupo5]
cd Acif104_Grupo5
```

## 2. Crear y activar el entorno virtual

El archivo `environment.yml` contiene todas las dependencias del proyecto.

### Paso 2.1 — Verificar que Conda está disponible

```bash
conda --version
```

En Windows, ejecutar estos comandos desde **Anaconda Prompt**. En VS Code, abrir la terminal
integrada con `Ctrl + Ñ` y seleccionar el perfil *Command Prompt* si PowerShell bloquea
la activación.

### Paso 2.2 — Crear el entorno a partir del archivo

```bash
conda env create -f environment.yml
```

Esto crea un entorno llamado **`ml`** con Python 3.12, pandas, numpy, scikit-learn 1.5,
xgboost, imbalanced-learn, shap, matplotlib, seaborn y jupyter. El proceso tarda entre
5 y 15 minutos según la conexión.

> Si la creación falla por la línea `prefix:` del archivo (una ruta local de un
> integrante), elimínala del `environment.yml`. Conda usará la ruta por defecto de cada equipo.

### Paso 2.3 — Activar el entorno

```bash
conda activate ml
```

El prompt debe mostrar el prefijo `(ml)`.

### Paso 2.4 — Instalar las dependencias del backend

```bash
pip install fastapi "uvicorn[standard]" joblib
```

### Paso 2.5 — Verificar la instalación

```bash
python -c "import pandas, sklearn, xgboost, shap, imblearn, fastapi; print('Entorno OK')"
```

### Comandos útiles del entorno

```bash
conda env list                                # Listar entornos disponibles
conda env update -f environment.yml --prune   # Actualizar tras cambios en el archivo
conda deactivate                              # Salir del entorno
conda env remove -n ml                        # Eliminar el entorno
```

## 3. Ejecutar el Jupyter Notebook

### Paso 3.1 — Registrar el kernel del entorno

Se hace una sola vez y permite que Jupyter y VS Code reconozcan el entorno `ml`:

```bash
conda activate ml
python -m ipykernel install --user --name ml --display-name "Python (ml)"
```

### Paso 3.2 — Opción A: Jupyter en el navegador

```bash
conda activate ml
jupyter notebook
```

Se abrirá `http://localhost:8888` en el navegador. Navegar a `notebooks/` y abrir
`Sumativa.ipynb`. En el menú **Kernel → Change Kernel** seleccionar **Python (ml)**.

Para JupyterLab:

```bash
jupyter lab
```

### Paso 3.3 — Opción B: Notebook dentro de VS Code

1. Instalar las extensiones **Python** y **Jupyter** de Microsoft.
2. Abrir la carpeta del proyecto (`File → Open Folder`).
3. Abrir `notebooks/Sumativa.ipynb`.
4. Pulsar **Select Kernel** (esquina superior derecha) → *Python Environments* → **Python (ml)**.

### Paso 3.4 — Ejecutar el análisis

Ejecutar las celdas en orden con **Run All** o `Shift + Enter` celda por celda.
El notebook espera encontrar el dataset en `data/ai4i2020.csv`.

Al finalizar, la última sección del notebook (serialización) genera los tres archivos
`.pkl` directamente en `backend/`, necesarios para que la API funcione.

**Tiempo estimado de ejecución completa:** 3–6 minutos (la celda de GridSearchCV es la
más lenta, entre 2 y 4 minutos según el equipo).

## 4. Levantar la API (backend)

```bash
conda activate ml
cd backend
uvicorn main:app --reload --port 8000
```

Al arrancar correctamente debe aparecer en consola:

```
[INFO] Artefactos cargados desde: C:\...\backend
INFO:     Application startup complete.
```

Documentación interactiva disponible en `http://127.0.0.1:8000/docs`.
Verificación de estado en `http://127.0.0.1:8000/health` (debe mostrar
`modelo_cargado`, `scaler_cargado` y `shap_disponible` en `true`).

Prueba rápida del endpoint:

```bash
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0,"type":"M"}'
```

## 5. Ejecutar el frontend

Con la API corriendo, abrir `frontend/index_1.html`. Se recomienda la extensión
**Live Server** de VS Code (clic derecho sobre el archivo → *Open with Live Server*)
para evitar restricciones del protocolo `file://`.

El frontend está configurado para consumir la API en `http://127.0.0.1:8000` mediante
la constante `API_URL` definida al inicio del `<script>`. Si la API corre en otro puerto
o equipo, ajustar esa constante.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10.000 registros, 14 columnas.

| Variable | Descripción | Uso |
|---|---|---|
| `UDI`, `Product ID` | Identificadores | Eliminadas (sin valor predictivo) |
| `Type` | Calidad del producto (L / M / H) | Predictora (One-Hot) |
| `Air temperature [K]` | Temperatura del aire | Predictora |
| `Process temperature [K]` | Temperatura del proceso | Predictora |
| `Rotational speed [rpm]` | Velocidad de rotación | Predictora |
| `Torque [Nm]` | Torque | Predictora |
| `Tool wear [min]` | Desgaste de herramienta | Predictora |
| `Machine failure` | Falla de máquina (0/1) | **Variable objetivo** |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Modos de falla específicos | Eliminadas (fuga de datos) |

Distribución de clases: 9.661 sin falla (96,61 %) / 339 con falla (3,39 %).

## Modelos implementados

- Random Forest (`n_estimators=100`)
- **XGBoost** (modelo seleccionado y servido en producción)
- Perceptrón Multicapa (64, 32, 16 neuronas)

## Técnicas aplicadas

Análisis exploratorio · One-Hot Encoding · StandardScaler · SMOTE · Submuestreo ·
GridSearchCV · SHAP (TreeExplainer)

## Métricas evaluadas

Accuracy · Precision · Recall · F1-Score · AUC-ROC (sobre probabilidades, no etiquetas) ·
Matriz de confusión · Curva ROC

## Tecnologías

Python 3.12 · pandas · NumPy · scikit-learn 1.5 · XGBoost · imbalanced-learn ·
SHAP · Matplotlib · Seaborn · FastAPI · Uvicorn · JavaScript · Chart.js

---

## Registro de implementación

Historial de los pasos realizados para pasar del prototipo con datos simulados
(Sumativa 1) a una plataforma funcional con predicciones reales (Sumativa 2).

### 1. Corrección del cálculo de AUC-ROC en el notebook

El notebook original calculaba `roc_auc_score()` usando las etiquetas predichas
(`y_pred`, valores 0/1) en lugar de las probabilidades. Esto reduce la curva ROC a
un único punto de operación y **subestima artificialmente** la capacidad discriminativa
del modelo.

- **Antes:** `roc_auc_score(y_test, y_pred_xgb)` → 0,8785
- **Después:** `roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])` → 0,978

Se corrigió tanto en la función `evaluar_modelo()` (reportes individuales) como en la
tabla comparativa de métricas, dejando ambas consistentes con el valor ya correcto que
mostraba la curva ROC.

### 2. Comparación de estrategias de balanceo de clases

Se agregó una nueva sección al notebook que entrena XGBoost bajo tres tratamientos del
desbalance (3,39 % de clase positiva), manteniendo fijo el conjunto de prueba:

1. **Sin balanceo** — mayor Precision, menor Recall.
2. **Submuestreo (`RandomUnderSampler`)** — mayor Recall, Precision degradada por
   pérdida de volumen de entrenamiento (de 6.763 a 237 registros de la clase mayoritaria).
3. **SMOTE** — mejor equilibrio general (F1-Score más alto), estrategia adoptada para
   el modelo final.

### 3. Ajuste de hiperparámetros con GridSearchCV

Se añadió una búsqueda exhaustiva sobre `n_estimators`, `max_depth`, `learning_rate`,
`subsample` y `colsample_bytree` (108 combinaciones × 5 folds de validación cruzada
estratificada), optimizando por F1-Score de la clase positiva. Se compararon los
resultados del modelo optimizado contra la línea base.

### 4. Serialización del modelo

Se agregó al final del notebook la celda que persiste los tres artefactos necesarios
para la inferencia en producción:

- `modelo_xgboost.pkl` — el modelo XGBoost entrenado.
- `scaler.pkl` — el `StandardScaler` ajustado (indispensable: el modelo fue entrenado
  sobre datos escalados, sin él las predicciones son inválidas).
- `columnas.pkl` — el orden exacto de las columnas esperado por el modelo.

**Incidente y corrección:** la primera versión de la celda usaba una ruta relativa
(`Path("../artifacts")`), lo que —dependiendo del directorio de trabajo del kernel—
terminó escribiendo los archivos fuera de la carpeta del proyecto. Se corrigió usando
una ruta absoluta explícita apuntando a `backend/`, y se añadió a la celda un bloque
de verificación que recarga el modelo guardado y confirma que reproduce exactamente
la misma predicción que el modelo en memoria.

### 5. Construcción de la API REST (`backend/main.py`)

Se creó desde cero el servicio FastAPI que reemplaza la lógica simulada del frontend:

- Carga el modelo, el escalador y el orden de columnas al iniciar.
- `POST /predict` — predicción individual: recibe las 5 variables de sensor + tipo de
  producto, aplica el mismo preprocesamiento del notebook (One-Hot + escalado) y
  devuelve probabilidad, clasificación de riesgo y el desglose de factores vía SHAP.
- `POST /predict-batch` — predicción por lotes, usada por la carga de CSV.
- `GET /health` — estado de disponibilidad del modelo, el escalador y SHAP.
- Validación de entrada con Pydantic (rangos físicamente plausibles por variable).
- CORS abierto para desarrollo local.

**Corrección posterior:** la carga de artefactos originalmente asumía una única ruta
fija (`backend/`), lo que rompía si el modelo aún no se había generado o estaba en otra
carpeta. Se reemplazó por una búsqueda en varias rutas candidatas (`backend/`,
`backend/artifacts/`, `../artifacts/`, raíz del proyecto), con un mensaje de error que
lista exactamente qué rutas se revisaron en caso de no encontrar el modelo.

### 6. Conexión del frontend con la API real

En `frontend/index_1.html` se reemplazó la lógica simulada de la pantalla de
predicción individual:

- **`PREDICT_FIELDS`**: el formulario pedía variables que el modelo nunca vio
  (temperatura en °C, vibración, presión, humedad). Se reemplazó por las 5 variables
  reales del dataset (temperatura del aire y del proceso en Kelvin, velocidad de
  rotación, torque, desgaste de herramienta), con rangos tomados del `describe()`
  del notebook.
- **Selector "Tipo de máquina"** → pasó a ser **"Calidad del producto (Type)"**,
  con las tres categorías reales del dataset (L / M / H).
- **`runPrediction()`**: se eliminó la fórmula de riesgo simulada
  (`score += Math.max(0,(values.vib-4))*7 …` con `setTimeout`) y se reemplazó por una
  función `async` que llama a `POST /predict` con `fetch`, incluyendo manejo de errores
  que activa el estado `predictErrorState` ya existente en el HTML (antes nunca se
  disparaba).

### 7. Depuración de errores post-integración

Durante las pruebas surgieron dos problemas, ambos resueltos:

- **Artefactos no encontrados (`RuntimeError` al iniciar Uvicorn):** causado por el
  incidente de ruta relativa del punto 4. Se resolvió regenerando los `.pkl` con la
  ruta absoluta corregida y actualizando `main.py` para buscar en múltiples ubicaciones.
- **Dashboard vacío / `Uncaught SyntaxError: Unexpected identifier 'document'`:**
  el reemplazo de `runPrediction()` quedó con una llave de cierre `}` de más después
  del `catch`, lo que rompía el parseo de todo el `<script>` — y por eso el
  `document.addEventListener('DOMContentLoaded', …)` que dispara el renderizado del
  panel principal (KPIs, gráfico de tendencia, alertas, tabla) nunca llegaba a
  registrarse. Se eliminó la llave sobrante.

### Estado actual

- Backend funcional, sirviendo predicciones reales desde el modelo XGBoost entrenado.
- Pantalla de "Predicción individual" conectada end-to-end a la API.
- Pendiente: conectar de la misma forma "Cargar archivo CSV" (`POST /predict-batch`),
  "Explicabilidad SHAP", "Monitoreo" e "Historial", que a la fecha siguen usando datos
  simulados generados en el propio `index_1.html`.

## Licencia

Proyecto académico. Dataset de uso público (UCI Machine Learning Repository).