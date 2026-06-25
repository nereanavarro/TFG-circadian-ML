import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # GEMINI

from scipy.signal import hilbert    # amplitud hilbert
from scipy.signal import find_peaks # parámetros - find_peaks "high/low_peak_stats"
from scipy import stats             # parámetros - tendencia  ""
from scipy.stats import pearsonr    # WL pearson coseno-fase 
import pyboat                       # info ondas

#=================================================
#      PROCESAMIENTO DE LOS DATOS                  ==> USAR: result = DataProcessing.pipeline(data)  => *data* ha de estar formato TABLA (no melt)
#=================================================
class DataProcessing:
    def pipeline(series, window=20):               # DETREND + ZSCORE

        series = series.apply(DataProcessing.detrend, ws=window)
        series = series.apply(DataProcessing.zscore)
        return series

    def detrend(series, ws=20):                              
        r =series.rolling(ws, center=True,min_periods=1).mean()
        return series-r
    
    def rolling_mean(df, cols, window=10):         # OTRA FORMA DE QUITAR LA TENDENCIA
        return df[cols]-df[cols].rolling(window=window, center=True, min_period=1).mean()
    
    def zscore (series):                          # (valor-media)desviación estandar --> mejor para comparar curvas cn distintas amplitudes
        mu = series.mean()                        # media
        sd = series.std()                         # desviación estándar
        return (series - mu ) / sd                # NOS PERMITE COMPARAR ENTRE EXPERIMENTOS, PONE VALORES AL REDEDOR DEL 0
        

class wl_period:
    def periodo_wavelet_pyboat(signal, fs=1):
        
        dt = 1 / fs  # espaciado entre muestras en horas
    
        # Rango de períodos a explorar (en horas)
        periodo_min = 2
        periodo_max = len(signal) // 3  # máximo 1/3 de la señal
    
        periodos = np.linspace(periodo_min, periodo_max, 300)
    
        # Calcular la transformada wavelet
        amplitudes, fases = pyboat.compute_spectrum(signal, dt, periodos)
    
        # Período con mayor amplitud media
        amplitud_media = np.mean(amplitudes, axis=1)
        idx_max = np.argmax(amplitud_media)
    
        return periodos[idx_max]


#================================================
#      INFORMACIÓN DE TABLAS-ONDAS      
#================================================
class info_tabla_ondas:
    def periodo_wavelet_pyboat(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        
        periodos = np.linspace(periodo_min, periodo_max, n_periodos)
        wAn = pyboat.WAnalyzer(periodos, fs_horas)
        wAn.compute_spectrum(signal, do_plot=False)
        ridge_data = wAn.get_maxRidge()
        return ridge_data.mean()                   # USAR ==>  resultados = dt_si.apply(lambda x: info_tabla_ondas.periodo_wavelet_pyboat(x))


# Wave ridge future extractions = sacar TODOS los datos posibles, no solo 5-6 de ahora; ver cuales son los q m interesan, por ej: amplitud/ power/ period



#==================================================
#      INFORMACIÓN DE TABLAS-ONDAS x PARÁMETROS     ==> USAR: result = pd.DataFrame(data.tolist(), index=param_data_treat.index)  => *data* ha de estar formato MELT 
#==================================================

class parametros_ondas:
    
    @staticmethod    # Para usar el método de ridge sin tener que mencionarlo sieeeempre en cada definición
    def analizar_onda(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):                     #   analisis_data = parametros_ondas.analizar_onda(data_melt.od)  => *data* formato MELT 
        periodos = np.linspace(periodo_min, periodo_max, n_periodos)
        wAn = pyboat.WAnalyzer(periodos, fs_horas)
        wAn.compute_spectrum(signal, do_plot=False)
        return wAn.get_maxRidge()
    
   # @staticmethod
    def period(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        ridge = parametros_ondas._analizar_onda(signal, fs_horas, periodo_min, periodo_max, n_periodos)
        return {"period": ridge['periods'].mean()}

   # @staticmethod
    def amplitude(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        ridge = parametros_ondas._analizar_onda(signal, fs_horas, periodo_min, periodo_max, n_periodos)
        return {"amplitude" : ridge['amplitude'].mean()}

   # @staticmethod
    def power(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        ridge = parametros_ondas._analizar_onda(signal, fs_horas, periodo_min, periodo_max, n_periodos)
        return {"power" : ridge['power'].mean()}

   # @staticmethod
    def phase(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        ridge = parametros_ondas._analizar_onda(signal, fs_horas, periodo_min, periodo_max, n_periodos)
        return {"phase" : ridge['phase'].mean()}

    @staticmethod
    def all(signal, fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        ridge = parametros_ondas.analizar_onda(signal, fs_horas, periodo_min, periodo_max, n_periodos)
        return {
            'period'   : ridge['periods'].mean(),
            'amplitude': ridge['amplitude'].mean(),
            'power'    : ridge['power'].mean(),
            'phase'    : ridge['phase'].mean()
                }

    #=================================================
    #                                                    ==> USAR (EN MELT): 1). Usar TODAS las condiciones result = parametros_ondas.get_peaks_stats(data_treat_ent)
    #   PARÁMETROS DE LOS DATOS (FIND_PEAKS)                                 2). Excluir condiciones (ej: quitar los Blank) 
                                                                                # conds_sin_blank = [c for c in data_treat_ent["condition"].unique() if "Blank" not in c]
    #=================================================                          # stats_df = get_peaks_stats(data_treat_ent, conditions=conds_sin_blank)

#=================================================
# EDITAMOS PARA QUE NOS DEVUELVA MÁS PARÁMETROS !!!!!! (1º CON + ASCENDENTES)

    def high_peaks_stats(data, conditions=None):
        """
        Detecta picos con find_peaks y devuelve un DataFrame con
        los parámetros para cada variable × condición.
    
        Parámetros
        ----------
        data        : pd.DataFrame con columnas [time, value, variable, condition]
        conditions  : lista de condiciones a analizar. Si es None, usa todas
                      las condiciones presentes en el DataFrame.
    
        Retorna
        -------
        pd.DataFrame : una fila por variable × condición con:
            - n_peaks             : número de picos detectados
            - mean/std_peak_value : altura media y dispersión de los picos
            - peak_to_peak        : diferencia entre el pico más alto y más bajo
            - mean_peak_time      : tiempo medio de los picos
            - mean/std_prominence : cuánto destaca el pico sobre su entorno
            - mean/std_width      : anchura media de los picos
            - mean_width_height   : altura a la que se mide la anchura
            - mean_left/right_ips : posición interpolada de los bordes del ancho
            - mean_left/right_thr : umbrales izquierdo/derecho del pico
            - mean_left/right_base: índice de la base de prominencia de cada lado
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()
    
        all_stats = []
    
        for cond in conditions:
            subset = data[data["condition"] == cond]
    
            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                values = var_df["value"].to_numpy()
                time   = var_df["time"].to_numpy()
    
                prom = np.abs(values).mean() * 1.5
    
                peaks, props = find_peaks(
                    values,
                    prominence  = prom,       # activa: prominences, left_bases, right_bases
                    width       = 0.1,        # activa: widths, width_heights, left_ips, right_ips
                    height      = -np.inf,       # activa: peak_heights
                    threshold   = 0,       # activa: left_thresholds, right_thresholds
                )
    
                if len(peaks) == 0:
                    continue
    
                peak_times  = time[peaks]
                peak_values = values[peaks]
    
                all_stats.append({
                    # ── Básicos ───────────────────────────────────────────────────
                    "condition":            cond,
                    "variable":             variable,
                    "n_peaks":              len(peaks),
                    "mean_peak_value":      peak_values.mean(),
                    "std_peak_value":       peak_values.std(),
                    "peak_to_peak":         peak_values.max() - peak_values.min(),
                    "mean_peak_time":       peak_times.mean(),
                    # ── Prominencia → requiere prominence= ───────────────────────
                    "mean_prominence":      props["prominences"].mean(),
                    "std_prominence":       props["prominences"].std(),
                    "mean_left_base":       props["left_bases"].mean(),
                    "mean_right_base":      props["right_bases"].mean(),
                    # ── Anchura → requiere width= ─────────────────────────────────
                    "mean_width":           props["widths"].mean(),
                    "std_width":            props["widths"].std(),
                    "mean_width_height":    props["width_heights"].mean(),
                    "mean_left_ips":        props["left_ips"].mean(),
                    "mean_right_ips":       props["right_ips"].mean(),
                    # ── Altura → requiere height= ─────────────────────────────────
                    "mean_peak_heights":    props["peak_heights"].mean(),
                    "std_peak_heights":     props["peak_heights"].std(),
                    # ── Umbrales → requiere threshold= ───────────────────────────
                    "mean_left_threshold":  props["left_thresholds"].mean(),
                    "mean_right_threshold": props["right_thresholds"].mean(),
                })
    
        return pd.DataFrame(all_stats)

#=================================================
            # PARA PICOS DESCENDENTES

    def low_peaks_stats(data, conditions=None):
        """
        Detecta picos con find_peaks y devuelve un DataFrame con
        los parámetros para cada variable × condición.
    
        Parámetros
        ----------
        data        : pd.DataFrame con columnas [time, value, variable, condition]
        conditions  : lista de condiciones a analizar. Si es None, usa todas
                      las condiciones presentes en el DataFrame.
    
        Retorna
        -------
        pd.DataFrame : una fila por variable × condición con:
            - n_peaks             : número de picos detectados
            - mean/std_peak_value : altura media y dispersión de los picos
            - peak_to_peak        : diferencia entre el pico más alto y más bajo
            - mean_peak_time      : tiempo medio de los picos
            - mean/std_prominence : cuánto destaca el pico sobre su entorno
            - mean/std_width      : anchura media de los picos
            - mean_width_height   : altura a la que se mide la anchura
            - mean_left/right_ips : posición interpolada de los bordes del ancho
            - mean_left/right_thr : umbrales izquierdo/derecho del pico
            - mean_left/right_base: índice de la base de prominencia de cada lado
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()
    
        all_stats = []
    
        for cond in conditions:
            subset = data[data["condition"] == cond]
    
            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                values = var_df["value"].to_numpy()
                time   = var_df["time"].to_numpy()
    
                prom = np.abs(values).mean() * 1.5
    
                peaks, props = find_peaks(
                    - values,
                    prominence  = prom,       # activa: prominences, left_bases, right_bases
                    width       = 0.1,        # activa: widths, width_heights, left_ips, right_ips
                    height      = -np.inf,       # activa: peak_heights
                    threshold   = 0,       # activa: left_thresholds, right_thresholds
                )
    
                if len(peaks) == 0:
                    continue
    
                peak_times  = time[peaks]
                peak_values = values[peaks]
    
                all_stats.append({
                    # ── Básicos ───────────────────────────────────────────────────
                    "condition":            cond,
                    "variable":             variable,
                    "n_peaks":              len(peaks),
                    "mean_peak_value":      peak_values.mean(),
                    "std_peak_value":       peak_values.std(),
                    "peak_to_peak":         peak_values.max() - peak_values.min(),
                    "mean_peak_time":       peak_times.mean(),
                    # ── Prominencia → requiere prominence= ───────────────────────
                    "mean_prominence":      props["prominences"].mean(),
                    "std_prominence":       props["prominences"].std(),
                    "mean_left_base":       props["left_bases"].mean(),
                    "mean_right_base":      props["right_bases"].mean(),
                    # ── Anchura → requiere width= ─────────────────────────────────
                    "mean_width":           props["widths"].mean(),
                    "std_width":            props["widths"].std(),
                    "mean_width_height":    props["width_heights"].mean(),
                    "mean_left_ips":        props["left_ips"].mean(),
                    "mean_right_ips":       props["right_ips"].mean(),
                    # ── Altura → requiere height= ─────────────────────────────────
                    "mean_peak_heights":    props["peak_heights"].mean(),
                    "std_peak_heights":     props["peak_heights"].std(),
                    # ── Umbrales → requiere threshold= ───────────────────────────
                    "mean_left_threshold":  props["left_thresholds"].mean(),
                    "mean_right_threshold": props["right_thresholds"].mean(),
                })
    
        return pd.DataFrame(all_stats)

#============================================

    def Low_peaks_stats(data, conditions=None):    # CON LOS PICOS ASCENDENTES !!!!
        """
        Detecta picos con find_peaks y devuelve un DataFrame con
        los parámetros para cada variable × condición.
    
        Parámetros
        ----------
        data        : pd.DataFrame con columnas [time, value, variable, condition]
        conditions  : lista de condiciones a analizar. Si es None, usa todas
                      las condiciones presentes en el DataFrame.
    
        Retorna
        -------
        pd.DataFrame : una fila por variable × condición
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()
    
        all_stats = []
    
        for cond in conditions:
            subset = data[data["condition"] == cond]
    
            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                values = var_df["value"].to_numpy()
                time   = var_df["time"].to_numpy()
    
                prom         = np.abs(values).mean() * 1.5
                peaks, props = find_peaks(- values, prominence=prom, width=0.1)
    
                if len(peaks) == 0:
                    continue
    
                peak_times  = time[peaks]
                peak_values = values[peaks]
    
                all_stats.append({
                    "condition":        cond,
                    "variable":         variable,
                    "n_peaks":          len(peaks),
                    "mean_prominence":  props["prominences"].mean(),
                    "std_prominence":   props["prominences"].std(),
                    "mean_width":       props["widths"].mean(),
                    "std_width":        props["widths"].std(),
                    "mean_peak_value":  peak_values.mean(),
                    "std_peak_value":   peak_values.std(),
                    "peak_to_peak":     peak_values.max() - peak_values.min(),
                    "mean_peak_time":   peak_times.mean(),
                })
    
        return pd.DataFrame(all_stats)


    #=================================================
    #                                                    ==> USAR (EN MELT): 1). Usar TODAS las condiciones result = parametros_ondas.get_peaks_stats(data_treat_ent)
    #   PARÁMETROS DE LOS DATOS ( TENDENCIA )                                2). Excluir condiciones (ej: quitar los Blank) 
                                                                                # conds_sin_blank = [c for c in data_treat_ent["condition"].unique() if "Blank" not in c]
    #=================================================                          # stats_df = get_peaks_stats(data_treat_ent, conditions=conds_sin_blank)
    
    def trend_stats(data, conditions=None, t_start=48, window=12):
        """
        Estudia la tendencia (positiva o negativa) de la señal a partir de t_start
        usando regresión lineal. Detecta si hay una tendencia sostenida y cuantifica
        cuánto cambia la señal respecto al valor inicial (t_start).
    
        Parámetros
        ----------
        data        : pd.DataFrame con columnas [time, value, variable, condition]   =====> EN FORMATO MELT !!!
        conditions  : lista de condiciones a analizar. Si es None, usa todas.
        t_start     : tiempo a partir del cual se analiza la tendencia (default: 48)
    
        Retorna:
        -------
        pd.DataFrame : una fila por variable × condición con:
            - slope                 : pendiente de la regresión (+ sube / - baja)
            - r_squared             : R² de la regresión (0-1, cuánto de lineal es la tendencia)
            - p_value               : significancia estadística de la tendencia
            - trend                 : 'increase' / 'decrease' / 'no trend' (si p > 0.05)
            - value_at_start        : valor medio de la señal en t_start
            - value_at_end          : valor medio de la señal al final
            - total_change          : diferencia absoluta (value_at_end - value_at_start)
            - pct_change            : % de cambio respecto al valor en t_start
            - trend_strength        : intensidad de la tendencia según R² → 'strong' (R²≥0.7) / 'moderate' (R²≥0.4) / 'weak' (R²<0.4) / 'no trend' (p≥0.05)
            - slope_normalized      : pendiente dividida por el valor inicial, permite comparar variables con distintas magnitudes (+ sube / - baja)
            - trend_start_time      : tiempo (hora) en el que empieza la tendencia sostenida,
                                       detectado cuando 3 ventanas consecutivas tienen pendiente
                                       significativa (p<0.05) y del mismo signo que la tendencia global
            - pct_change            : % de cambio desde trend_start_time hasta el final de la señal,
            - pct_change_from_trend : % .... indica cuánto cambia la señal una vez que la tendencia ya es clara

        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()
    
        all_stats = []
    
        for cond in conditions:
            subset = data[data["condition"] == cond]
    
            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
    
                mask   = var_df["time"] >= t_start
                seg    = var_df[mask]
    
                if len(seg) < window:
                    continue
    
                values = seg["value"].to_numpy()
                time   = seg["time"].to_numpy()
    
                # ── Regresión lineal global ───────────────────────────────────────
                slope, intercept, r, p_value, _ = stats.linregress(time, values)
                r_squared      = r ** 2
                value_at_start = values[:3].mean()
                value_at_end   = values[-3:].mean()
                total_change   = value_at_end - value_at_start
                pct_change     = (total_change / np.abs(value_at_start)) * 100 if value_at_start != 0 else np.nan
    
                # ── Pendiente normalizada ─────────────────────────────────────────
                slope_normalized = slope / np.abs(value_at_start) if value_at_start != 0 else np.nan
    
                # ── Trend strength ────────────────────────────────────────────────
                if p_value >= 0.05:
                    trend          = "no trend"
                    trend_strength = "no trend"
                else:
                    trend = "increase" if slope > 0 else "decrease"
                    if r_squared >= 0.7:
                        trend_strength = "strong"
                    elif r_squared >= 0.4:
                        trend_strength = "moderate"
                    else:
                        trend_strength = "weak"
    
                # ── Ventana deslizante: inicio de tendencia ───────────────────────
                # Calcula la pendiente en cada ventana y busca cuándo se vuelve
                # consistentemente del mismo signo que la tendencia global
                trend_start_time = np.nan
                window_slopes    = []
    
                for i in range(len(values) - window + 1):
                    w_time   = time[i: i + window]
                    w_values = values[i: i + window]
                    w_slope, _, _, w_p, _ = stats.linregress(w_time, w_values)
                    window_slopes.append((time[i], w_slope, w_p))
    
                # Buscar primera ventana donde la pendiente es significativa
                # y del mismo signo que la tendencia global, y se mantiene 3 ventanas seguidas
                if trend != "no trend":
                    for i in range(len(window_slopes) - 2):
                        t0, s0, p0 = window_slopes[i]
                        _, s1, p1  = window_slopes[i + 1]
                        _, s2, p2  = window_slopes[i + 2]
    
                        same_sign   = all(np.sign(s) == np.sign(slope) for s in [s0, s1, s2])
                        significant = all(p < 0.05 for p in [p0, p1, p2])
    
                        if same_sign and significant:
                            trend_start_time = t0
                            break
    
                # % cambio desde inicio tendencia hasta el final
                if not np.isnan(trend_start_time):
                    val_at_trend_start = values[time == trend_start_time]
                    pct_change_from_trend = (
                        (value_at_end - val_at_trend_start[0]) / np.abs(val_at_trend_start[0]) * 100
                        if len(val_at_trend_start) > 0 and val_at_trend_start[0] != 0
                        else np.nan
                    )
                else:
                    pct_change_from_trend = np.nan
    
                all_stats.append({
                    "condition":              cond,
                    "variable":               variable,
                    "trend":                  trend,
                    "trend_strength":         trend_strength,
                    "slope":                  slope,
                    "slope_normalized":       slope_normalized,
                    "r_squared":              r_squared,
                    "p_value":                p_value,
                    "value_at_start":         value_at_start,
                    "value_at_end":           value_at_end,
                    "total_change":           total_change,
                    "pct_change":             pct_change,
                    "trend_start_time":       trend_start_time,
                    "pct_change_from_trend":  pct_change_from_trend,
                })
    
        return pd.DataFrame(all_stats)




    # ─────────────────────────────────────────────────────────────────────────
    #   PARÁMETROS WAVELET (fase, amplitud, power)  – en bucle desde el dbm
    #   ==> USAR (EN MELT, igual que high_peaks_stats):
    #       wave_params = parametros_ondas.wave_params(data_treat_ent)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def wave_params(data, conditions=None,
                    fs_horas=1, periodo_min=2, periodo_max=30, n_periodos=300):
        """
        Extrae fase media, amplitud media y power medio de la transformada
        wavelet para cada variable × condición del DataFrame en formato MELT.

        Parámetros
        ----------
        data        : pd.DataFrame con columnas [time, value, variable, condition]
        conditions  : lista de condiciones a analizar. Si es None, usa todas.
        fs_horas    : frecuencia de muestreo en horas (default 1)
        periodo_min : período mínimo en horas (default 2)
        periodo_max : período máximo en horas (default 30)
        n_periodos  : resolución del análisis wavelet (default 300)

        Retorna
        -------
        pd.DataFrame con columnas:
            - condition, variable
            - WV_phase     : fase media (radianes) del ridge de máxima amplitud
            - WV_amplitude : amplitud media del ridge
            - WV_power     : power medio del ridge
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()

        all_stats = []

        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy()

                # ── Ajuste automático del período máximo ──────────────────────
                periodo_max_real = min(periodo_max, len(signal) // 3)
                if periodo_max_real < periodo_min:
                    continue   # señal demasiado corta, no tiene sentido analizarla

                try:
                    ridge = parametros_ondas.analizar_onda(
                        signal, fs_horas, periodo_min, periodo_max_real, n_periodos
                    )
                    all_stats.append({
                        "condition"    : cond,
                        "variable"     : variable,
                        "WV_phase"     : ridge["phase"].mean(),
                        "WV_amplitude" : ridge["amplitude"].mean(),
                        "WV_power"     : ridge["power"].mean(),
                    })
                except Exception:
                    continue   # sin ridge claro

        return pd.DataFrame(all_stats)

# ─────────────────────────────────────────────────────────────────────────
    #   COSENO DE LA FASE + MÉTRICAS: Pearson, Spearman, Cross-corr, DTW
    #   ==> USAR (EN MELT):
    #       cos_df = parametros_ondas.phase_cosine_similarity(data_treat_ent)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def phase_cosine_similarity(data, conditions=None,
                                fs_horas=1, periodo_min=2,
                                periodo_max=30, n_periodos=300):
        """
        Para cada variable × condición extrae la serie temporal de fase
        instantánea (ridge wavelet) y calcula cuatro métricas de relación
        entre fase y cos(fase):

            - WV_cos_phase_mean            : media del coseno de la fase
            - WV_pearson_r / WV_pearson_p  : correlación lineal
            - WV_spearman_r / WV_spearman_p: correlación de rangos (no lineal)
            - WV_crosscorr_lag             : lag en horas de máxima cross-corr
            - WV_crosscorr_max             : valor máximo de la cross-corr
            - WV_dtw_distance              : distancia DTW (series normalizadas)
        """
        try:
            from dtaidistance import dtw as dtw_lib
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "dtaidistance", "--quiet"])
            from dtaidistance import dtw as dtw_lib

        from scipy.stats import pearsonr, spearmanr

        if conditions is None:
            conditions = data["condition"].unique().tolist()

        all_stats = []

        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy()

                # ── Ajuste automático del período máximo ──────────────────────
                periodo_max_real = min(periodo_max, len(signal) // 3)
                if periodo_max_real < periodo_min:
                    continue   # señal demasiado corta

                try:
                    ridge = parametros_ondas.analizar_onda(
                        signal, fs_horas, periodo_min, periodo_max_real, n_periodos
                    )
                    phase_series = ridge["phase"].to_numpy()
                    cos_series   = np.cos(phase_series)

                    # ── Pearson ───────────────────────────────────────────────
                    r_pe, p_pe = pearsonr(phase_series, cos_series)

                    # ── Spearman ──────────────────────────────────────────────
                    r_sp, p_sp = spearmanr(phase_series, cos_series)

                    # ── Cross-correlation ─────────────────────────────────────
                    ph_n  = (phase_series - phase_series.mean()) / (phase_series.std() + 1e-10)
                    cos_n = (cos_series   - cos_series.mean())   / (cos_series.std()   + 1e-10)
                    xcorr = np.correlate(ph_n, cos_n, mode="full")
                    lags  = np.arange(-len(phase_series) + 1, len(phase_series))
                    lag_max   = int(lags[np.argmax(xcorr)])
                    xcorr_max = float(xcorr.max() / len(phase_series))

                    # ── DTW ───────────────────────────────────────────────────
                    ph_n_64  = ph_n.astype(np.float64)
                    cos_n_64 = cos_n.astype(np.float64)
                    dtw_dist = dtw_lib.distance_fast(ph_n_64, cos_n_64)

                    all_stats.append({
                        "condition"         : cond,
                        "variable"          : variable,
                        "WV_cos_phase_mean" : cos_series.mean(),
                        "WV_pearson_r"      : r_pe,
                        "WV_pearson_p"      : p_pe,
                        "WV_spearman_r"     : r_sp,
                        "WV_spearman_p"     : p_sp,
                        "WV_crosscorr_lag"  : lag_max,
                        "WV_crosscorr_max"  : xcorr_max,
                        "WV_dtw_distance"   : dtw_dist,
                    })
                except Exception:
                    continue

        return pd.DataFrame(all_stats)
    # ─────────────────────────────────────────────────────────────────────────
    #   DERIVADAS DE LA SEÑAL (1ª, 2ª, 3ª) + ESTADÍSTICOS + ASIMETRÍA
    #   ==> USAR (EN MELT):
    #       deriv_df = parametros_ondas.derivative_params(data_treat_ent)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def derivative_params(data, conditions=None):
        """
        Para cada variable × condición calcula la 1ª, 2ª y 3ª derivada numérica
        de la señal tratada (detrend+zscore) usando np.gradient, y extrae:

        Derivadas (prefijo D1_, D2_, D3_):
            - _abs_max  : valor absoluto máximo (velocidad/aceleración máxima)
            - _abs_mean : media del valor absoluto (velocidad/aceleración promedio)
            - _std      : desviación estándar (variabilidad)

        Asimetría subida/bajada (de la 1ª derivada):
            - D1_asym_mean : media del ratio velocidad_subida / velocidad_bajada
                             entre todos los ciclos detectados.
                             >1 → sube rápido y baja lento
                             <1 → sube lento y baja rápido
                             ~1 → subida y bajada similares
            - D1_asym_std  : desviación estándar del ratio entre ciclos

        Parámetros
        ----------
        data       : pd.DataFrame [time, value, variable, condition]
        conditions : lista de condiciones; None = todas.

        Retorna
        -------
        pd.DataFrame con las columnas anteriores + condition, variable.
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()

        all_stats = []

        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy()
                time   = var_df["time"].to_numpy()

                if len(signal) < 6:
                    continue

                try:
                    # ── Derivadas numéricas ───────────────────────────────────
                    d1 = np.gradient(signal, time)   # velocidad de cambio
                    d2 = np.gradient(d1,     time)   # aceleración
                    d3 = np.gradient(d2,     time)   # tasa de cambio de aceleración

                    # ── Estadísticos por derivada ─────────────────────────────
                    def deriv_stats(d):
                        abs_d = np.abs(d)
                        return abs_d.max(), abs_d.mean(), d.std()

                    d1_max, d1_mean, d1_std = deriv_stats(d1)
                    d2_max, d2_mean, d2_std = deriv_stats(d2)
                    d3_max, d3_mean, d3_std = deriv_stats(d3)

                    # ── Asimetría subida/bajada ───────────────────────────────
                    # Localizar picos de la señal original (D1 cruza cero de + a -)
                    # Un pico ocurre cuando d1 pasa de positivo a negativo
                    # Un valle ocurre cuando d1 pasa de negativo a positivo
                    # Para cada ciclo pico-a-pico calculamos:
                    #   velocidad_subida = media(d1 > 0) entre valle y pico
                    #   velocidad_bajada = media(|d1 < 0|) entre pico y siguiente valle

                    # Detectar picos (máximos) y valles (mínimos) con find_peaks
                    prom_thr = np.abs(signal).mean() * 0.5
                    peaks,  _ = find_peaks( signal, prominence=prom_thr)
                    valleys,_ = find_peaks(-signal, prominence=prom_thr)

                    asym_ratios = []

                    for pk in peaks:
                        # Valle anterior más cercano
                        prev_valleys = valleys[valleys < pk]
                        next_valleys = valleys[valleys > pk]

                        if len(prev_valleys) == 0 or len(next_valleys) == 0:
                            continue

                        v_prev = prev_valleys[-1]
                        v_next = next_valleys[0]

                        # Velocidad media de subida (d1 entre valle prev y pico)
                        seg_up   = d1[v_prev:pk]
                        # Velocidad media de bajada (|d1| entre pico y valle next)
                        seg_down = d1[pk:v_next]

                        v_up   = seg_up[seg_up > 0].mean()   if len(seg_up[seg_up > 0])   > 0 else np.nan
                        v_down = np.abs(seg_down[seg_down < 0]).mean() if len(seg_down[seg_down < 0]) > 0 else np.nan

                        if not np.isnan(v_up) and not np.isnan(v_down) and v_down > 0:
                            asym_ratios.append(v_up / v_down)

                    asym_mean = np.mean(asym_ratios)  if len(asym_ratios) > 0 else np.nan
                    asym_std  = np.std(asym_ratios)   if len(asym_ratios) > 1 else np.nan

                    all_stats.append({
                        "condition"    : cond,
                        "variable"     : variable,
                        # 1ª derivada
                        "D1_abs_max"   : d1_max,
                        "D1_abs_mean"  : d1_mean,
                        "D1_std"       : d1_std,
                        # 2ª derivada
                        "D2_abs_max"   : d2_max,
                        "D2_abs_mean"  : d2_mean,
                        "D2_std"       : d2_std,
                        # 3ª derivada
                        "D3_abs_max"   : d3_max,
                        "D3_abs_mean"  : d3_mean,
                        "D3_std"       : d3_std,
                        # Asimetría
                        "D1_asym_mean" : asym_mean,
                        "D1_asym_std"  : asym_std,
                    })

                except Exception:
                    continue

        return pd.DataFrame(all_stats)


    # ─────────────────────────────────────────────────────────────────────────
    #   HORA CIRCADIANA DE LOS PICOS DEL COSENO DE LA FASE
    #   ==> USAR (EN MELT):
    #       cos_peaks_df = parametros_ondas.cos_phase_peak_time(data_treat_ent)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def cos_phase_peak_time(data, conditions=None,
                            fs_horas=1, periodo_min=2,
                            periodo_max=30, n_periodos=300,
                            period_norm=24):
        """
        Para cada variable × condición:
          1. Extrae la serie temporal del coseno de la fase (ridge wavelet).
          2. Detecta los PICOS (máximos) del cos(fase) con find_peaks.
          3. Normaliza cada hora de pico a un ciclo de period_norm horas
             (hora_pico mod period_norm → hora circadiana, ej: 37h → 13h del ciclo).
          4. Devuelve la media de esas horas circadianas y el nº de picos.

        Un pico del cos(fase) corresponde a un momento donde la fase pasa por 0,
        es decir, cuando la señal está en su MÁXIMO. La hora circadiana media
        indica a qué hora del ciclo de 24h tiende a ocurrir ese máximo.

        Parámetros
        ----------
        data        : pd.DataFrame [time, value, variable, condition]
        conditions  : lista de condiciones; None = todas.
        period_norm : periodo de normalización en horas (default 24)

        Retorna
        -------
        pd.DataFrame con columnas:
            - condition, variable
            - COS_n_peaks          : número de picos detectados en cos(fase)
            - COS_peak_phase_mean  : media de (hora_pico mod 24), hora circadiana
                                     media del pico del coseno de la fase
            - COS_peak_phase_std   : dispersión de esas horas entre ciclos
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()

        all_stats = []

        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy()
                time   = var_df["time"].to_numpy()

                periodo_max_real = min(periodo_max, len(signal) // 3)
                if periodo_max_real < periodo_min:
                    continue

                try:
                    ridge       = parametros_ondas.analizar_onda(
                        signal, fs_horas, periodo_min, periodo_max_real, n_periodos
                    )
                    phase_series = ridge["phase"].to_numpy()
                    cos_series   = np.cos(phase_series)

                    # Tiempo recortado al tamaño del ridge
                    t_ridge = time[:len(cos_series)]

                    # ── Detectar picos del cos(fase) ──────────────────────────
                    # Prominencia mínima = 0.3 (cos va de -1 a +1, picos reales
                    # destacan claramente sobre los valles)
                    peaks, _ = find_peaks(cos_series, prominence=0.3)

                    if len(peaks) == 0:
                        continue

                    peak_hours = t_ridge[peaks]               # horas absolutas de cada pico

                    # ── Normalizar a ciclo de 24h ─────────────────────────────
                    peak_circadian = peak_hours % period_norm  # hora dentro del ciclo

                    all_stats.append({
                        "condition"           : cond,
                        "variable"            : variable,
                        "COS_n_peaks"         : len(peaks),
                        "COS_peak_phase_mean" : peak_circadian.mean(),
                        "COS_peak_phase_std"  : peak_circadian.std(),
                    })

                except Exception:
                    continue

        return pd.DataFrame(all_stats)


    # ─────────────────────────────────────────────────────────────────────────
    #   PARÁMETROS DE RITMICIDAD DEL FREE RUN
    #   ==> USAR (EN MELT, sobre data_treat_fr):
    #       fr_df = parametros_ondas.freerun_params(data_treat_fr)
    #
    #   Columnas generadas (prefijo FR_):
    #     FR_wv_period      : período medio del ridge wavelet (horas)
    #     FR_wv_amplitude   : amplitud media wavelet en el free run
    #     FR_wv_power       : power medio wavelet en el free run
    #     FR_ls_period      : período dominante por Lomb-Scargle (horas)
    #     FR_ls_power       : potencia del pico dominante en Lomb-Scargle
    #     FR_ls_pvalue      : p-valor del test de ritmicidad (Lomb-Scargle)
    #     FR_rhythmic       : True/False — rítmico si FR_ls_pvalue < alpha
    #     FR_rac            : Relative Amplitude Coefficient = (max-min)/(max+min)
    #                         rango 0-1; cuanto mayor, más rítmica la señal
    #     FR_amp_decay      : ratio amplitud_último_tercio / amplitud_primer_tercio
    #                         cercano a 1 = self-sustained; << 1 = amortiguado
    #     FR_is_sustained   : True si FR_amp_decay >= amp_decay_threshold (default 0.5)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    # ─────────────────────────────────────────────────────────────────────────
    #   PARÁMETROS DE RITMICIDAD DEL FREE RUN  (v7)
    #   ==> USAR (sobre data_treat_fr, t >= 125h):
    #       fr_df = parametros_ondas.freerun_params(data_treat_fr)
    #
    #   Columnas generadas (prefijo FR_):
    #     FR_wv_period_w   : período medio ponderado por power (wavelet)
    #     FR_hilbert_amp   : amplitud media envolvente de Hilbert
    #     FR_hilbert_cv    : CV del envelope = std/mean
    #                        ≈ 0 → amplitud constante (self-sustained)
    #                        alto → amplitud variable (no sostenida)
    #     FR_rac           : Relative Amplitude con envelope Hilbert
    #                        = (max-min)/(|max|+|min|) del envelope
    #     FR_autocorr_lag  : lag (h) del primer pico de autocorrelación
    #                        en ventana 16-32h → estimación del período
    #     FR_autocorr_val  : altura de ese pico → fuerza de la periodicidad
    #     FR_ls_pval       : p-valor Lomb-Scargle corregido (Scargle 1982)
    #     FR_ls_period     : período dominante por Lomb-Scargle (h)
    #     FR_rhythmic      : True/False según FR_ls_pval < alpha (0.05)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def freerun_params(data, conditions=None,
                       fs_horas=1, periodo_min=16, periodo_max=32, n_periodos=300,
                       autocorr_min=16, autocorr_max=32,
                       alpha=0.05):
        """
        Extrae parametros de ritmicidad del free run para cada variable x condicion.

        Parametros
        ----------
        data             : pd.DataFrame [time, value, variable, condition]
                           data_treat_fr (señal tratada, t >= 125h)
        conditions       : lista de condiciones; None = todas.
        fs_horas         : frecuencia de muestreo en horas (default 1)
        periodo_min/max  : rango wavelet en horas (default 16-32)
        autocorr_min/max : ventana busqueda pico autocorrelacion (horas)
        alpha            : umbral p-valor para FR_rhythmic (default 0.05)

        Retorna
        -------
        pd.DataFrame con columnas FR_ descritas arriba.
        """
        from scipy.signal import hilbert, find_peaks, lombscargle

        if conditions is None:
            conditions = data["condition"].unique().tolist()

        all_stats = []

        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy().astype(float)
                time   = var_df["time"].to_numpy().astype(float)

                if len(signal) < 48:
                    continue

                # ── Imputar NaN (ultimo timepoint siempre NaN en el instrumento)
                if np.isnan(signal).any():
                    s_ser  = pd.Series(signal)
                    signal = s_ser.interpolate(method="linear").ffill().bfill().to_numpy()

                if np.isnan(signal).all() or signal.std() == 0:
                    continue

                periodo_max_real = min(periodo_max, len(signal) // 3)
                if periodo_max_real < periodo_min:
                    continue

                try:
                    # ══════════════════════════════════════════════════════════
                    # 1. PERIODO PONDERADO POR POWER (wavelet)
                    #    numpy.average(periods, weights=power) da mas peso a
                    #    los instantes donde la oscilacion es mas fuerte
                    # ══════════════════════════════════════════════════════════
                    ridge = parametros_ondas.analizar_onda(
                        signal, fs_horas, periodo_min, periodo_max_real, n_periodos
                    )
                    fr_wv_period_w = float(np.average(
                        ridge["periods"], weights=ridge["power"]
                    ))

                    # ══════════════════════════════════════════════════════════
                    # 2. AMPLITUD DE HILBERT + CV + RAC
                    #    hilbert() → señal analitica; su modulo = envolvente
                    #    instantanea de amplitud
                    # ══════════════════════════════════════════════════════════
                    envelope = np.abs(hilbert(signal))

                    fr_hilbert_amp = float(envelope.mean())
                    fr_hilbert_cv  = float(envelope.std() / envelope.mean())                                      if envelope.mean() > 0 else np.nan

                    # RAC sobre la envolvente Hilbert
                    e_max  = envelope.max()
                    e_min  = envelope.min()
                    denom  = abs(e_max) + abs(e_min)
                    fr_rac = float((e_max - e_min) / denom) if denom > 0 else np.nan

                    # ══════════════════════════════════════════════════════════
                    # 3. AUTOCORRELACION
                    #    Normalizada por N. Busca el primer pico entre
                    #    autocorr_min y autocorr_max horas.
                    #    lag  → estimacion del periodo dominante
                    #    val  → fuerza de la periodicidad (0=sin ritmo, 1=perfecto)
                    # ══════════════════════════════════════════════════════════
                    n        = len(signal)
                    sig_z    = (signal - signal.mean()) / (signal.std() + 1e-10)
                    acf_full = np.correlate(sig_z, sig_z, mode="full") / n
                    acf      = acf_full[n - 1:]          # lags 0..N-1

                    search   = acf[autocorr_min : autocorr_max + 1]
                    peaks, _ = find_peaks(search, height=0)

                    if len(peaks) > 0:
                        best_pk         = peaks[np.argmax(search[peaks])]
                        fr_autocorr_lag = float(autocorr_min + best_pk)
                        fr_autocorr_val = float(search[best_pk])
                    else:
                        # sin pico claro: maximo del rango
                        idx_max         = int(np.argmax(search))
                        fr_autocorr_lag = float(autocorr_min + idx_max)
                        fr_autocorr_val = float(search[idx_max])

                    # ══════════════════════════════════════════════════════════
                    # 4. RITMICIDAD — Lomb-Scargle corregido (Scargle 1982)
                    #    normalize=True devuelve potencia en [0,1]
                    #    conversion a z sin normalizar: z = pn * (N/2)
                    #    p = 1 - (1 - exp(-z))^M
                    # ══════════════════════════════════════════════════════════
                    periods_ls = np.linspace(periodo_min, periodo_max_real, 500)
                    freqs_ang  = 2 * np.pi / periods_ls
                    sig_norm   = signal - signal.mean()

                    pgram    = lombscargle(
                        time.astype(float), sig_norm.astype(float),
                        freqs_ang, normalize=True
                    )
                    idx_peak      = np.argmax(pgram)
                    fr_ls_period  = float(periods_ls[idx_peak])
                    pn            = float(pgram[idx_peak])
                    z_unnorm      = pn * (len(signal) / 2.0)
                    M             = len(freqs_ang)
                    fr_ls_pval    = float(1.0 - (1.0 - np.exp(-z_unnorm)) ** M)
                    fr_rhythmic   = bool(fr_ls_pval < alpha)

                    all_stats.append({
                        "condition"       : cond,
                        "variable"        : variable,
                        "FR_wv_period_w"  : fr_wv_period_w,
                        "FR_hilbert_amp"  : fr_hilbert_amp,
                        "FR_hilbert_cv"   : fr_hilbert_cv,
                        "FR_rac"          : fr_rac,
                        "FR_autocorr_lag" : fr_autocorr_lag,
                        "FR_autocorr_val" : fr_autocorr_val,
                        "FR_ls_pval"      : fr_ls_pval,
                        "FR_ls_period"    : fr_ls_period,
                        "FR_rhythmic"     : fr_rhythmic,
                    })

                except Exception:
                    continue

        return pd.DataFrame(all_stats)
    # ════════════════════════════════════════════════════════════════════════
    # NUEVA FUNCIÓN v9 — metacycle_params (ritmicidad MetaCycle via R)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def metacycle_params(data,
                         rscript_path="Rscript",
                         r_script="run_meta2d.R",
                         t_col="time",
                         val_col="value",
                         id_col="variable",
                         cond_col="condition"):
        """
        Calcula ritmicidad con meta2d (MetaCycle, R) sobre un DataFrame en
        formato LONG (típicamente data_treat_fr, t>=125h).

        Workflow:
          1. Pivot long → wide (filas=pocillos, columnas=timepoints)
          2. Guarda input.txt temporal con primera línea como timepoints
          3. Llama Rscript run_meta2d.R input.txt output_dir/
          4. Lee meta2d_result.csv y devuelve DataFrame con:
             [condition, variable, MC_period, MC_pvalue, MC_BH_Q]

        Parámetros
        ----------
        data : DataFrame LONG con columnas [time, variable, value, condition]
        rscript_path : ruta al ejecutable Rscript.
            Windows típico: "C:/Program Files/R/R-4.4.x/bin/Rscript.exe"
            Linux/Mac:      "Rscript" (si está en el PATH)
        r_script : ruta al script run_meta2d.R

        Devuelve
        --------
        DataFrame con columnas [condition, variable, MC_period, MC_pvalue, MC_BH_Q]
        — UNA fila por pocillo. El cálculo de MC_rhythmic (booleano) se hace
        después en el notebook a partir de MC_BH_Q < 0.001 (o el umbral elegido).
        """
        import os, tempfile, subprocess
        import pandas as pd

        # ── 1. LONG → WIDE: filas=pocillos, columnas=timepoints ─────────────
        wide = (data
                .pivot_table(index=[cond_col, id_col],
                             columns=t_col, values=val_col)
                .reset_index())

        # Mapping (cond, var) para reunir tras procesar
        mapping = wide[[cond_col, id_col]].copy()

        # Formato para R: primera col = CycID (nombre pocillo), resto = valores
        wide_r = wide.drop(columns=[cond_col]).rename(columns={id_col: "CycID"})
        time_cols = sorted([c for c in wide_r.columns if c != "CycID"])
        wide_r = wide_r[["CycID"] + time_cols]

        # ── 2. Guardar input.txt + crear output_dir temporales ──────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                          mode="w") as tmp:
            wide_r.to_csv(tmp.name, sep="\t", index=False)
            input_path = tmp.name
        output_dir = tempfile.mkdtemp()

        try:
            # ── 3. Llamar Rscript run_meta2d.R ────────────────────────────────
            result = subprocess.run(
                [rscript_path, r_script, input_path, output_dir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print("⚠️ R STDERR:\n", result.stderr)
                print("⚠️ R STDOUT:\n", result.stdout)
                raise RuntimeError(
                    f"Rscript falló (returncode={result.returncode}). "
                    f"Revisa que R y MetaCycle estén instalados y la ruta "
                    f"rscript_path sea correcta."
                )

            # ── 4. Leer resultado ────────────────────────────────────────────
            out_csv = os.path.join(output_dir, "meta2d_result.csv")
            out_df = pd.read_csv(out_csv)

        finally:
            # ── 5. Limpieza temporales ───────────────────────────────────────
            try:
                os.remove(input_path)
            except OSError:
                pass
            try:
                for f in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, f))
                os.rmdir(output_dir)
            except OSError:
                pass

        # ── 6. Renombrar columnas ───────────────────────────────────────────
        # meta2d a veces aplana las columnas con prefijo 'meta.' (cuando devuelve
        # una lista de DataFrames en vez de uno solo). Buscamos cada columna
        # primero sin prefijo, luego con prefijo 'meta.'
        col_map = {}
        wanted = [("CycID",         "variable"),
                  ("meta2d_period", "MC_period"),
                  ("meta2d_pvalue", "MC_pvalue"),
                  ("meta2d_BH.Q",   "MC_BH_Q")]
        for old, new in wanted:
            if old in out_df.columns:
                col_map[old] = new
            elif f"meta.{old}" in out_df.columns:
                col_map[f"meta.{old}"] = new
            else:
                raise KeyError(
                    f"No encuentro la columna '{old}' ni 'meta.{old}' en el "
                    f"output de meta2d. Columnas disponibles: "
                    f"{list(out_df.columns)}"
                )
        out_df = out_df.rename(columns=col_map)

        # ── 7. Reunir cond+var con resultados ───────────────────────────────
        result_df = mapping.merge(
            out_df[["variable", "MC_period", "MC_pvalue", "MC_BH_Q"]],
            on="variable", how="left"
        )
        return result_df


    # ════════════════════════════════════════════════════════════════════════
    # NUEVA FUNCIÓN v10 — fr_peaks_stats (picos en free run, 2 criterios)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def fr_peaks_stats(data, conditions=None,
                       prom_factor_mean=1.5,
                       prom_factor_max=0.20):
        """
        Detecta picos en la ventana de FREE RUN (t >= 125 h) con DOS criterios
        de prominencia y devuelve 6 columnas por pocillo. Es la réplica de
        HP_n_peaks / HP_mean_peak_value / HP_std_peak_value (que se calculaban
        sobre el entrainment) ahora aplicada al free run, con dos versiones
        del umbral para poder comparar.

        Parámetros
        ----------
        data             : pd.DataFrame [time, value, variable, condition]
                           típicamente data_treat_fr (señal tratada, t >= 125)
        conditions       : lista de condiciones a analizar; None = todas
        prom_factor_mean : multiplicador para el criterio A — prominencia
                           umbral = prom_factor_mean * mean(|signal|).
                           Default 1.5 (igual que high_peaks_stats).
        prom_factor_max  : multiplicador para el criterio B — prominencia
                           umbral = prom_factor_max * max(|signal|).
                           Default 0.20 (20 % del máximo).

        Retorna
        -------
        pd.DataFrame con columnas:
            condition, variable,
            FR_n_peaks,          FR_mean_peak_value,    FR_std_peak_value
            FR_n_peaks_02,       FR_mean_peak_value_02, FR_std_peak_value_02

        Notas
        -----
        - Los pocillos sin picos detectados reciben valor 0 en n_peaks y 0
          en mean/std (en vez de NaN), siguiendo la preferencia de tener
          números válidos para ML/estadística.
        - Los NaN puntuales de la señal (p. ej. último timepoint) se imputan
          linealmente antes de detectar picos.
        """
        if conditions is None:
            conditions = data["condition"].unique().tolist()

        def _peaks_one(signal, prom_thr):
            """Devuelve (n, mean, std) o (0, 0, 0) si no hay picos."""
            try:
                pk_idx, _ = find_peaks(signal, prominence=prom_thr)
                if len(pk_idx) > 0:
                    pv = signal[pk_idx]
                    return int(len(pk_idx)), float(pv.mean()), float(pv.std())
                return 0, 0.0, 0.0
            except Exception:
                return 0, 0.0, 0.0

        all_stats = []
        for cond in conditions:
            subset = data[data["condition"] == cond]

            for variable, var_df in subset.groupby("variable"):
                var_df = var_df.sort_values("time").reset_index(drop=True)
                signal = var_df["value"].to_numpy().astype(float)

                # ── Sanity checks ────────────────────────────────────────────
                if len(signal) < 5:
                    continue
                if np.isnan(signal).any():
                    signal = (pd.Series(signal)
                              .interpolate(method="linear").ffill().bfill()
                              .to_numpy())
                if np.isnan(signal).all() or signal.std() == 0:
                    # Señal plana: 0 picos por definición
                    all_stats.append({
                        "condition"             : cond,
                        "variable"              : variable,
                        "FR_n_peaks"            : 0,
                        "FR_mean_peak_value"    : 0.0,
                        "FR_std_peak_value"     : 0.0,
                        "FR_n_peaks_02"         : 0,
                        "FR_mean_peak_value_02" : 0.0,
                        "FR_std_peak_value_02"  : 0.0,
                    })
                    continue

                # ── Criterio A — prom = prom_factor_mean * mean(|signal|) ────
                prom_A = float(np.abs(signal).mean() * prom_factor_mean)
                n_A, m_A, s_A = _peaks_one(signal, prom_A)

                # ── Criterio B — prom = prom_factor_max * max(|signal|) ──────
                prom_B = float(np.max(np.abs(signal)) * prom_factor_max)
                n_B, m_B, s_B = _peaks_one(signal, prom_B)

                all_stats.append({
                    "condition"             : cond,
                    "variable"              : variable,
                    "FR_n_peaks"            : n_A,
                    "FR_mean_peak_value"    : m_A,
                    "FR_std_peak_value"     : s_A,
                    "FR_n_peaks_02"         : n_B,
                    "FR_mean_peak_value_02" : m_B,
                    "FR_std_peak_value_02"  : s_B,
                })

        return pd.DataFrame(all_stats)

    # ════════════════════════════════════════════════════════════════════════
    # NUEVA FUNCIÓN v10 — classify_fr_params (categorización ordinal)
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def classify_fr_params(df,
                           period_col='FR_wv_period_w',
                           n_peaks_col='FR_n_peaks',
                           autocorr_col='FR_autocorr_val',
                           rhythmic_col='MC_rhythmic'):
        """
        Clasifica 4 parámetros de ritmicidad en categorías con etiquetas de
        texto (string) descriptivas, añadiéndolas como columnas nuevas al
        DataFrame. Los nombres de columnas nuevas siguen el patrón
        '{prefijo}c_{nombre_original}' donde la 'c' marca "categórica".

        ──────────────────────────────────────────────────────────────────────
        FRc_wv_period_w   ← FR_wv_period_w  (período wavelet, umbrales fijos)
            "muy corto"   [16, 20) h
            "corto"       [20, 24) h
            "largo"       [24, 28) h
            "muy largo"   [28, 32] h
        ──────────────────────────────────────────────────────────────────────
        FRc_n_peaks       ← FR_n_peaks      (picos, umbrales fijos)
            "ninguno/muy pocos"   (0-1 picos)
            "pocos"               (2-3 picos)
            "bien"                (4-6 picos)
            "muchos"              (7-10 picos)
            "ruido"               (>= 11 picos)
        ──────────────────────────────────────────────────────────────────────
        FRc_autocorr_val  ← FR_autocorr_val (autocorrelación, PERCENTILES)
            "bajo"        FR_autocorr_val <  Q1        (25% inferior)
            "medio-bajo"  FR_autocorr_val in [Q1, Q2)
            "medio-alto"  FR_autocorr_val in [Q2, Q3)
            "alto"        FR_autocorr_val >= Q3        (25% superior)
            Etiquetas relativas a la propia distribución del dataset
            (Q1, Q2, Q3 se imprimen al ejecutar).
        ──────────────────────────────────────────────────────────────────────
        MCc_rhythmic      ← MC_rhythmic     (booleano → string)
            "Falso"       MC_rhythmic == 0
            "Verdadero"   MC_rhythmic == 1
        ──────────────────────────────────────────────────────────────────────

        Parámetros
        ----------
        df            : pd.DataFrame con las columnas indicadas
        period_col    : columna del período (default 'FR_wv_period_w')
        n_peaks_col   : columna del nº picos (default 'FR_n_peaks')
        autocorr_col  : columna de autocorrelación (default 'FR_autocorr_val')
        rhythmic_col  : columna boolean de ritmicidad (default 'MC_rhythmic')

        Devuelve
        --------
        Copia del DataFrame con las 4 nuevas columnas de texto añadidas.

        Si alguna columna de entrada no existe, se omite su clasificación
        correspondiente sin error y se imprime un aviso.
        """
        result = df.copy()

        # ── 1. PERÍODO → FRc_wv_period_w ─────────────────────────────────────
        if period_col in result.columns:
            def _period_label(p):
                if pd.isna(p):  return pd.NA
                if p < 20:      return "muy corto"
                if p < 24:      return "corto"
                if p < 28:      return "largo"
                return "muy largo"
            result['FRc_wv_period_w'] = (
                result[period_col].map(_period_label).astype('string')
            )
        else:
            print(f"⚠️  classify_fr_params: '{period_col}' no existe — "
                  f"omito FRc_wv_period_w")

        # ── 2. Nº DE PICOS → FRc_n_peaks ─────────────────────────────────────
        if n_peaks_col in result.columns:
            def _peaks_label(n):
                if pd.isna(n):  return pd.NA
                n = int(n)
                if n <= 1:      return "ninguno/muy pocos"
                if n <= 3:      return "pocos"
                if n <= 6:      return "bien"
                if n <= 10:     return "muchos"
                return "ruido"
            result['FRc_n_peaks'] = (
                result[n_peaks_col].map(_peaks_label).astype('string')
            )
        else:
            print(f"⚠️  classify_fr_params: '{n_peaks_col}' no existe — "
                  f"omito FRc_n_peaks")

        # ── 3. AUTOCORRELACIÓN → FRc_autocorr_val (PERCENTILES + texto) ──────
        # Etiquetas relativas a la propia distribución del dataset
        # (Q1, Q2, Q3). Se usa percentiles porque los umbrales fijos de
        # literatura (0.2, 0.4, 0.6) no encajan con señales detrended+z-scored.
        if autocorr_col in result.columns:
            vals = result[autocorr_col].dropna()
            if len(vals) == 0:
                print(f"⚠️  classify_fr_params: '{autocorr_col}' todos NaN — "
                      f"omito FRc_autocorr_val")
            else:
                q1, q2, q3 = vals.quantile([0.25, 0.50, 0.75]).values
                def _ac_label(v):
                    if pd.isna(v):  return pd.NA
                    if v < q1:      return "bajo"
                    if v < q2:      return "medio-bajo"
                    if v < q3:      return "medio-alto"
                    return "alto"
                result['FRc_autocorr_val'] = (
                    result[autocorr_col].map(_ac_label).astype('string')
                )
                print(f"📊 FRc_autocorr_val: percentiles de "
                      f"'{autocorr_col}' aplicados → "
                      f"Q1={q1:.3f}  Q2={q2:.3f}  Q3={q3:.3f}")
        else:
            print(f"⚠️  classify_fr_params: '{autocorr_col}' no existe — "
                  f"omito FRc_autocorr_val")

        # ── 4. RITMICIDAD MC → MCc_rhythmic ──────────────────────────────────
        if rhythmic_col in result.columns:
            def _rh_label(v):
                if pd.isna(v):  return pd.NA
                return "Verdadero" if int(v) == 1 else "Falso"
            result['MCc_rhythmic'] = (
                result[rhythmic_col].map(_rh_label).astype('string')
            )
        else:
            print(f"⚠️  classify_fr_params: '{rhythmic_col}' no existe — "
                  f"omito MCc_rhythmic")

        return result
