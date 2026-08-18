# Perfectime — Mantenimiento Predictivo de Equipos Industriales

Proyecto para la asignatura **Aprendizaje de Máquina (ACIF104)** — Universidad Andrés Bello.

**Repositorio:** https://github.com/XetuRiot/Acif104_Grupo5

comando de clonacion git clone 'https://github.com/XetuRiot/Acif104_Grupo5'

## Integrantes

Javiera Chávez · Joaquín Olivares · Claudio Yáñez


## 1. Crear el entorno

```bash
conda env create -f environment.yml
conda activate ml
pip install fastapi "uvicorn[standard]" joblib
```

> Si falla por la línea `prefix:` del `environment.yml`, bórrala y vuelve a intentar.

## 2. Entrenar el modelo
comando 'cd Acif104_Grupo5
jupyter notebook notebooks/Sumativa.ipynb'

Abrir `notebooks/Sumativa.ipynb` con el kernel del entorno `ml` y ejecutar todas las celdas (**Run All**). Al terminar quedan generados `modelo_xgboost.pkl`, `scaler.pkl` y `columnas.pkl` en `modelos/`.

> Para probar el notebook con otro dataset (sin editar el código), definir la variable de entorno `PERFECTIME_DATASET_PATH` con la ruta del CSV antes de abrir Jupyter. Si no se define, se usa `datasets/ai4i2020.csv` por defecto.

## 3. Levantar la API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Documentación interactiva en `http://127.0.0.1:8000/docs`, estado del servicio en `http://127.0.0.1:8000/health`.

## 4. Abrir el frontend

Con la API corriendo, abrir `frontend/index_1.html` (recomendado con la extensión **Live Server** de VS Code para evitar restricciones de `file://`). Todas las pantallas usan datos reales de la API.

'se debe abrir el backend y el frontend en 2 terminales distintas para funcionar'

## 5. Reentrenar el modelo con datos nuevos

Desde la pantalla **Monitoreo** se puede subir un CSV con registros históricos que incluyan el resultado real (columna `fallo_real`, 0 o 1) para que el modelo se reentrene con esos datos además de `ai4i2020.csv`. Se necesitan al menos 10 registros nuevos. Los registros se van acumulando en `datasets/nuevos_registros.csv` (no se versiona) y el modelo (`modelos/*.pkl`) se actualiza automáticamente, sin reiniciar la API.

## Licencia

Proyecto académico. Dataset de uso público (UCI Machine Learning Repository).
