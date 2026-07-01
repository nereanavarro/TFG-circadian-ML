# Understanding circadian gene expression through Machine Learning

TFG de Nerea Navarro. Análisis del entrainment circadiano en Bacillus subtilis (cepa ytvA).

## Aviso importante: caso de uso, sin datos

Este repositorio es un **caso de uso**: documenta la metodología y presenta los
resultados obtenidos, pero **no incluye ningún dato** con el que se ha trabajado
(ni datos crudos de luminiscencia, ni datasets procesados/intermedios en Excel).
Esta es una restriccion impuesta por el director del proyecto.

Como consecuencia, **los notebooks no se pueden ejecutar de principio a fin por
terceros**: el código se publica para transparencia metodológica (qué se hizo y
cómo), no como un pipeline reproducible de forma autónoma. Las salidas de celda
también se han eliminado de los notebooks por el mismo motivo. Los resultados
agregados se presentan únicamente como figuras (`results/figuras/`) y como
informe narrativo sin cifras (`results/informe.md`).

## Estructura

- `src/ma_methods10.py`: módulo con las funciones de procesado y ML.
- `notebooks/`: pipeline numerado de 01 a 03 (sin salidas de celda), mas
  `notebooks/run_meta2d.R` (script R que integra MetaCycle, ver metodología).
- `results/`: figuras generadas e informe de resultados, sin tablas de datos.

## Requisitos técnicos

- **Python 3.10 o superior** (compatible desde Python 3.9, por las versiones
  mínimas de `pandas`/`numpy` fijadas en `requirements.txt`). Descarga en
  https://www.python.org/downloads/. Dependencias con
  `pip install -r requirements.txt` (se recomienda un entorno virtual).
- **R 4.2 o superior** (https://cran.r-project.org/) con el paquete `MetaCycle`
  instalado (`install.packages("MetaCycle")`). Solo lo necesita el notebook 01,
  para el bloque de ritmicidad estadística vía `notebooks/run_meta2d.R`.

## Metodología (resumen)

1. **01 — Extraccion de parametros**: consolida los experimentos de
   luminiscencia en un dataset unico, calculando parametros de picos,
   tendencia, wavelet, derivadas, fase de coseno y ritmicidad estadistica
   (MetaCycle/R: JTK_CYCLE + ARS + Lomb-Scargle). Requiere R instalado con el
   paquete MetaCycle.
2. **02 — Clasificación**: diez modelos de clasificacion (AdaBoost,
   RandomForest, GradientBoosting, ExtraTrees, SVM, KNN, regresión logística,
   MLP, XGBoost, LightGBM) para predecir `reporter`, `genome` y `medium` a
   partir de los parámetros extraidos.
3. **03 — Predicción multi-output**: extiende el problema a la prediccion
   simultanea de seis descriptores de ritmicidad en fase de free-run
   (`MultiOutputClassifier`), con seleccion de variables por informacion
   mutua, validacion cruzada y analisis SHAP de interpretabilidad.

Ver `results/informe.md` para una descripción cualitativa de los resultados.

## Autora

Nerea Navarro Valcárcel · TFG · Supervisor: Borja Ferrero Bordera.
