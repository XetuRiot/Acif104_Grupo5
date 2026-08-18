# Perfectime — Mantenimiento Predictivo de Equipos Industriales

Proyecto para la asignatura **Aprendizaje de Máquina (ACIF104)** — Universidad Andrés Bello.

**Repositorio:** https://github.com/XetuRiot/Acif104_Grupo5

## Integrantes

Javiera Chávez · Joaquín Olivares · Claudio Yáñez

## Clonar el repositorio

```bash
git clone https://github.com/XetuRiot/Acif104_Grupo5
cd Acif104_Grupo5
```

---

## Método A — Git Bash

1. Crear el entorno (una sola vez):
   ```bash
   conda env create -f environment.yml
   conda activate ml
   pip install fastapi "uvicorn[standard]" joblib
   ```
   > Si falla por la línea `prefix:` del `environment.yml`, bórrala y vuelve a intentar.

2. Entrenar el modelo (una sola vez):
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/Sumativa.ipynb
   ```
   Al terminar quedan generados `modelo_xgboost.pkl`, `scaler.pkl` y `columnas.pkl` en `modelos/`.
   > Para usar otro dataset, definir `PERFECTIME_DATASET_PATH` con la ruta del CSV antes de este paso. Si no se define, se usa `datasets/ai4i2020.csv`.

3. Levantar la API. Abrir una terminal de Git Bash y activar el entorno:
   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate ml
   cd backend
   uvicorn main:app --reload --port 8000
   ```

4. Levantar el frontend. Abrir **otra** terminal de Git Bash, activar el entorno igual que en el paso 3, y correr:
   ```bash
   cd frontend
   python -m http.server 5500
   ```
   Abrir en el navegador `http://127.0.0.1:5500/index_1.html`.
   > Si el puerto 5500 está ocupado, usar otro: `python -m http.server 5501`.

---

## Método B — VS Code

1. Abrir la carpeta del proyecto en VS Code.

2. Crear el entorno (una sola vez), en una terminal integrada (`Ctrl+ñ`):
   ```bash
   conda env create -f environment.yml
   conda activate ml
   pip install fastapi "uvicorn[standard]" joblib
   ```
   > Si falla por la línea `prefix:` del `environment.yml`, bórrala y vuelve a intentar.

3. Entrenar el modelo (una sola vez): abrir `notebooks/Sumativa.ipynb`, elegir el kernel `ml` (arriba a la derecha) y ejecutar **Run All**. Al terminar quedan generados `modelo_xgboost.pkl`, `scaler.pkl` y `columnas.pkl` en `modelos/`.
   > Para usar otro dataset, definir `PERFECTIME_DATASET_PATH` con la ruta del CSV antes de abrir el notebook. Si no se define, se usa `datasets/ai4i2020.csv`.

4. Levantar la API, en una terminal integrada con el entorno `ml` activado:
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

5. Levantar el frontend: clic derecho sobre `frontend/index_1.html` → **"Open with Live Server"**. Se abre en `http://127.0.0.1:5500/index_1.html`.
   > Live Server vigila toda la carpeta del proyecto, así que puede recargar la página automáticamente cuando el backend guarda una predicción nueva en `historial.db`, cortando la vista del resultado. Si pasa, basta con volver a mirar la pantalla (los datos ya quedaron guardados).

---

Con cualquiera de los dos métodos, todas las pantallas usan datos reales de la API. El backend y el frontend deben quedar corriendo al mismo tiempo, en terminales distintas.

## Reentrenar el modelo con datos nuevos

Desde la pantalla **Monitoreo** se puede subir un CSV con registros históricos que incluyan el resultado real (columna `fallo_real`, 0 o 1) para que el modelo se reentrene con esos datos además de `ai4i2020.csv`. Se necesitan al menos 10 registros nuevos. Los registros se van acumulando en `datasets/nuevos_registros.csv` (no se versiona) y el modelo (`modelos/*.pkl`) se actualiza automáticamente, sin reiniciar la API.

## Licencia

Proyecto académico. Dataset de uso público (UCI Machine Learning Repository).
