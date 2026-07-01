# Informe de resultados

Este informe resume, de forma cualitativa y sin mostrar cifras, tablas ni ningún
dato con el que se ha trabajado, los resultados obtenidos al ejecutar de
principio a fin los tres notebooks del pipeline (`01_parametros_lumi_dbm10`,
`02_Machine_Learning_v8`, `03_ML_v11_FR_multioutput`) en un entorno de
desarrollo privado, con datos que no se distribuyen en este repositorio. Las
figuras generadas están en `results/figuras/`.

## Notebook 01 — Pipeline de extracción de parámetros

Consolida varios experimentos de luminiscencia (`lumi_*`) en un único dataset,
calculando parámetros de picos, tendencia, wavelet, derivadas, fase de coseno y
ritmicidad estadística (MetaCycle/R). Se ejecutó de principio a fin sin errores,
incluyendo el bloque de MetaCycle, y el dataset consolidado resultante se
verificó frente a una versión de referencia previa, confirmando que el
pipeline es reproducible fielmente a partir de los datos crudos. Este notebook
no genera figuras, solo la tabla consolidada que alimenta a los notebooks 02 y
03.

## Notebook 02 — Clasificación (reporter / genome / medium)

Se entrenaron diez modelos de clasificación para predecir tres variables categóricas
a partir de los parámetros extraídos en el notebook 01. Los modelos basados en
ensembles de árboles (ExtraTrees, LightGBM, RandomForest, XGBoost, GradientBoosting)
obtuvieron un desempeño claramente superior y consistente en las tres variables
objetivo, mientras que los modelos basados en distancia o lineales (SVM, KNN,
regresión logística) y la red neuronal (MLP) quedaron notablemente por detrás,
en algunos casos cercanos al azar. ExtraTrees fue el modelo más sólido en general.
Las figuras incluyen matrices de confusión y ranking de importancia de variables
para el modelo ganador, además de curvas ROC comparativas entre todos los modelos.

## Notebook 03 — Predicción multi-output (6 targets de ritmicidad)

Extiende el problema a la predicción simultánea de seis descriptores de ritmicidad
en fase de free-run, un problema mucho más difícil que el del notebook 02 por el
crecimiento combinatorio del espacio de salida (el propio notebook incluye un
análisis de esta complejidad frente a un baseline aleatorio). De nuevo, los modelos
basados en árboles (ExtraTrees en cabeza, seguido de cerca por KNN, RandomForest y
GradientBoosting) superaron con claridad a SVM, regresión logística y MLP.

Aplicar selección de variables por información mutua (quedándose con el subconjunto
más informativo) y validar con validación cruzada de 5 particiones mejoró el
rendimiento medio de casi todos los modelos basados en árboles, mientras que
empeoró ligeramente a los modelos ya débiles (lineales y MLP) — indicio de que la
reducción de variables ayuda sobre todo a los modelos capaces de aprovecharla.

El análisis SHAP sobre el modelo ganador muestra que los targets no comparten
exactamente las mismas variables más influyentes, lo que sugiere que los seis
descriptores de ritmicidad no son completamente redundantes entre sí, sino que
capturan aspectos algo distintos de la dinámica circadiana.

Con un dataset de partida reducido, varias clases dentro de algunos targets
están poco representadas; el propio notebook cuantifica este desbalance y se
observa, de forma consistente en todos los modelos, un recall claramente más
débil en las clases minoritarias. Esta limitación debe tenerse en cuenta al
interpretar cualquier cifra de accuracy agregada.

## Incidencias relevantes encontradas y resueltas durante el proceso

- **requirements.txt incompleto**: faltaban `pyboat`, `xgboost` y
  `dtaidistance` (este último se auto-instalaba desde el propio código, lo
  cual no es buena práctica). Se han añadido a `requirements.txt`. La
  dependencia en R + paquete MetaCycle no es instalable vía pip y se ha
  documentado aparte en el README.
- **Fix aplicado en el notebook 03**: por un cambio de comportamiento de
  pandas ≥3.0 (`astype(str)` ya no convierte los `NaN` en la cadena `"nan"`),
  una fila con un valor de target sin etiquetar rompía el informe de
  validación final. Se añadió una celda que elimina esa fila antes de
  codificar los targets, documentado en el propio notebook.
