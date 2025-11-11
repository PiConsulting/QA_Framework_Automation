import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# === CONFIGURACIÓN ===
ENDPOINTS = {
    "gpt-4o": "http://localhost:7071/api/private-assistant"
}

CONFIG = {
    "archivo_entrada": "archivo.xlsx",
    "timeout": 60,
    "reintentos": 2,
    "delay_entre_requests": 0.5,
    "delay_entre_modelos": 2.0  # Pausa entre terminar un modelo y empezar el siguiente
}


def cargar_preguntas(archivo: str) -> pd.DataFrame:
    """Carga el archivo de preguntas (Excel o CSV)"""
    archivo_path = Path(archivo)

    if not archivo_path.exists():
        raise FileNotFoundError(f"❌ No se encuentra el archivo: {archivo}")

    if archivo.endswith(".csv"):
        df = pd.read_csv(archivo)
    elif archivo.endswith((".xlsx", ".xls")):
        df = pd.read_excel(archivo)
    else:
        raise ValueError("❌ Formato no soportado. Usa .csv, .xlsx o .xls")

    if "pregunta" not in df.columns:
        raise ValueError(
            "⚠️ El archivo debe tener una columna llamada 'pregunta'")

    return df


def enviar_pregunta(
    pregunta: str,
    modelo: str,
    reintentos: int = 2,
    timeout: int = 60
) -> Dict:
    """Envía una pregunta al endpoint y retorna el resultado"""

    if modelo not in ENDPOINTS:
        return {
            "respuesta": f"Error: Modelo '{modelo}' no existe",
            "status": None,
            "duracion": 0,
            "error": True
        }

    url = ENDPOINTS[modelo]
    payload = {"question": pregunta}
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    for intento in range(reintentos + 1):
        inicio = time.time()
        resp = None

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            duracion = round(time.time() - inicio, 2)

            if resp.status_code == 200:
                data = resp.json()
                respuesta = data.get("answer") or data.get(
                    "response") or str(data)

                return {
                    "respuesta": respuesta,
                    "status": resp.status_code,
                    "duracion": duracion,
                    "error": False
                }
            else:
                if intento < reintentos:
                    print(f"      ⚠️ Reintento {intento + 1}/{reintentos}...")
                    time.sleep(1)
                    continue

                return {
                    "respuesta": f"Error {resp.status_code}: {resp.text[:200]}",
                    "status": resp.status_code,
                    "duracion": duracion,
                    "error": True
                }

        except requests.exceptions.Timeout:
            duracion = round(time.time() - inicio, 2)
            if intento < reintentos:
                print(
                    f"      ⏱️ Timeout, reintentando {intento + 1}/{reintentos}...")
                time.sleep(2)
                continue

            return {
                "respuesta": f"Timeout después de {timeout}s",
                "status": None,
                "duracion": duracion,
                "error": True
            }

        except Exception as e:
            duracion = round(time.time() - inicio, 2)
            if intento < reintentos:
                print(
                    f"      🔄 Error, reintentando {intento + 1}/{reintentos}...")
                time.sleep(2)
                continue

            return {
                "respuesta": f"Exception: {str(e)}",
                "status": resp.status_code if resp else None,
                "duracion": duracion,
                "error": True
            }


def procesar_preguntas_comparativo(df: pd.DataFrame) -> List[Dict]:
    """Procesa todas las preguntas con TODOS los modelos - MODELO POR MODELO"""

    modelos = ["gpt-4o"]
    total_preguntas = len(df)

    print("\n" + "=" * 80)
    print("🚀 PROCESAMIENTO COMPARATIVO - SECUENCIAL POR MODELO")
    print("=" * 80)
    print(f"📊 Total de preguntas: {total_preguntas}")
    print(f"🤖 Modelos: {', '.join(modelos)}")
    print(f"📈 Total de consultas: {total_preguntas * len(modelos)}")
    print("=" * 80 + "\n")

    # Inicializar diccionario para almacenar resultados por pregunta
    resultados_por_pregunta = {}

    # Inicializar estructura de resultados
    for i, row in df.iterrows():
        pregunta = str(row["pregunta"]).strip()
        if pregunta and pregunta.lower() != "nan":
            resultados_por_pregunta[i] = {
                "pregunta": pregunta,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    # PROCESAR MODELO POR MODELO
    for idx_modelo, modelo in enumerate(modelos, 1):
        print("\n" + "🔹" * 40)
        print(f"🤖 PROCESANDO MODELO: {modelo} ({idx_modelo}/{len(modelos)})")
        print("🔹" * 40 + "\n")

        for i, row in df.iterrows():
            pregunta = str(row["pregunta"]).strip()

            if not pregunta or pregunta.lower() == "nan":
                continue

            print(f"  [{i+1}/{total_preguntas}] 📝 {pregunta[:50]}...", end=" ")

            resultado = enviar_pregunta(
                pregunta,
                modelo,
                reintentos=CONFIG["reintentos"],
                timeout=CONFIG["timeout"]
            )

            # Guardar resultado en la estructura
            resultados_por_pregunta[i][f"respuesta_{modelo}"] = resultado["respuesta"]
            resultados_por_pregunta[i][f"response_time_{modelo}"] = resultado["duracion"]
            resultados_por_pregunta[i][f"status_{modelo}"] = resultado["status"]

            # Mostrar resultado
            status_emoji = "✅" if not resultado["error"] else "❌"
            print(f"{status_emoji} {resultado['duracion']}s")

            # Pausa entre preguntas del mismo modelo
            if i < total_preguntas - 1:
                time.sleep(CONFIG["delay_entre_requests"])

        print(f"\n✅ Modelo {modelo} completado")

        # Pausa entre modelos
        if idx_modelo < len(modelos):
            print(
                f"⏳ Esperando {CONFIG['delay_entre_modelos']}s antes del siguiente modelo...\n")
            time.sleep(CONFIG["delay_entre_modelos"])

    # Convertir a lista de diccionarios
    resultados = list(resultados_por_pregunta.values())

    # Calcular estadísticas comparativas para cada pregunta
    for resultado in resultados:
        tiempos = {}
        for modelo in modelos:
            tiempo_key = f"response_time_{modelo}"
            if tiempo_key in resultado and resultado[tiempo_key] > 0:
                tiempos[modelo] = resultado[tiempo_key]

        if tiempos:
            modelo_mas_rapido = min(tiempos, key=tiempos.get)
            modelo_mas_lento = max(tiempos, key=tiempos.get)

            resultado["modelo_mas_rapido"] = modelo_mas_rapido
            resultado["modelo_mas_lento"] = modelo_mas_lento
            resultado["tiempo_mas_rapido"] = tiempos[modelo_mas_rapido]
            resultado["tiempo_mas_lento"] = tiempos[modelo_mas_lento]
            resultado["diferencia_max_seg"] = round(
                tiempos[modelo_mas_lento] - tiempos[modelo_mas_rapido], 2
            )

    return resultados


def guardar_resultados(resultados: List[Dict], archivo_entrada: str):
    """Guarda los resultados comparativos en Excel"""

    if not resultados:
        print("⚠️ No hay resultados para guardar")
        return

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = Path(archivo_entrada).stem
    archivo_salida = f"comparativa_{nombre_base}_{timestamp}.xlsx"

    # Crear DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Definir orden de columnas
    modelos = ["gpt-4o", "gpt-4o-mini",
               "gpt-4.1", "gpt-4.1-mini", "gpt-5-nano"]

    columnas_orden = ["pregunta"]

    # Agregar columnas por modelo
    for modelo in modelos:
        columnas_orden.extend([
            f"respuesta_{modelo}",
            f"response_time_{modelo}",
            f"status_{modelo}"
        ])

    # Agregar columnas de comparación
    columnas_orden.extend([
        "modelo_mas_rapido",
        "tiempo_mas_rapido",
        "modelo_mas_lento",
        "tiempo_mas_lento",
        "diferencia_max_seg",
        "timestamp"
    ])

    # Usar solo las columnas que existen
    columnas_finales = [
        col for col in columnas_orden if col in df_resultados.columns]
    df_resultados = df_resultados[columnas_finales]

    # Guardar con formato
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        # Hoja principal
        df_resultados.to_excel(writer, index=False, sheet_name='Comparativa')

        # Hoja de resumen estadístico
        df_stats = calcular_estadisticas_globales(resultados, modelos)
        df_stats.to_excel(writer, index=False, sheet_name='Estadísticas')

        # Ajustar ancho de columnas - Hoja Comparativa
        worksheet = writer.sheets['Comparativa']
        for idx, col in enumerate(df_resultados.columns, 1):
            max_length = max(
                df_resultados[col].astype(str).map(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(
                64 + idx)].width = min(max_length + 2, 60)

    # Mostrar estadísticas en consola
    mostrar_estadisticas(resultados, modelos, archivo_salida)


def calcular_estadisticas_globales(resultados: List[Dict], modelos: List[str]) -> pd.DataFrame:
    """Calcula estadísticas globales por modelo"""

    stats = []

    for modelo in modelos:
        tiempo_key = f"response_time_{modelo}"
        status_key = f"status_{modelo}"

        tiempos = [r[tiempo_key]
                   for r in resultados if tiempo_key in r and r[tiempo_key] > 0]
        exitos = sum(1 for r in resultados if r.get(status_key) == 200)
        total = len(resultados)

        victorias = sum(1 for r in resultados if r.get(
            "modelo_mas_rapido") == modelo)

        stats.append({
            "modelo": modelo,
            "preguntas_exitosas": exitos,
            "preguntas_fallidas": total - exitos,
            "tasa_exito_%": round(exitos / total * 100, 2) if total > 0 else 0,
            "tiempo_promedio_seg": round(sum(tiempos) / len(tiempos), 2) if tiempos else 0,
            "tiempo_minimo_seg": round(min(tiempos), 2) if tiempos else 0,
            "tiempo_maximo_seg": round(max(tiempos), 2) if tiempos else 0,
            "tiempo_total_seg": round(sum(tiempos), 2) if tiempos else 0,
            "veces_mas_rapido": victorias
        })

    return pd.DataFrame(stats)


def mostrar_estadisticas(resultados: List[Dict], modelos: List[str], archivo: str):
    """Muestra estadísticas en consola"""

    total = len(resultados)

    print("\n" + "=" * 80)
    print(f"✅ Resultados guardados en: {archivo}")
    print("\n📊 ESTADÍSTICAS COMPARATIVAS POR MODELO")
    print("=" * 80)

    for modelo in modelos:
        tiempo_key = f"response_time_{modelo}"
        status_key = f"status_{modelo}"

        tiempos = [r[tiempo_key]
                   for r in resultados if tiempo_key in r and r[tiempo_key] > 0]
        exitos = sum(1 for r in resultados if r.get(status_key) == 200)
        victorias = sum(1 for r in resultados if r.get(
            "modelo_mas_rapido") == modelo)

        promedio = sum(tiempos) / len(tiempos) if tiempos else 0

        print(f"\n🤖 {modelo.upper()}")
        print(f"   • Exitosas: {exitos}/{total} ({exitos/total*100:.1f}%)")
        print(f"   • Tiempo promedio: {promedio:.2f}s")
        print(f"   • Tiempo total: {sum(tiempos):.2f}s")
        print(f"   • Veces más rápido: {victorias}")

    # Modelo ganador general
    ganadores = {}
    for r in resultados:
        ganador = r.get("modelo_mas_rapido")
        if ganador:
            ganadores[ganador] = ganadores.get(ganador, 0) + 1

    if ganadores:
        modelo_ganador = max(ganadores, key=ganadores.get)
        print(f"\n🏆 MODELO MÁS RÁPIDO GENERAL: {modelo_ganador.upper()}")
        print(
            f"   Ganó en {ganadores[modelo_ganador]}/{total} preguntas ({ganadores[modelo_ganador]/total*100:.1f}%)")

    print("=" * 80)


def main():
    """Función principal"""
    try:
        # Cargar preguntas
        print(f"\n📂 Cargando archivo: {CONFIG['archivo_entrada']}")
        df = cargar_preguntas(CONFIG["archivo_entrada"])
        print(f"✅ {len(df)} preguntas cargadas")

        # Procesar preguntas con todos los modelos
        resultados = procesar_preguntas_comparativo(df)

        # Guardar resultados
        guardar_resultados(resultados, CONFIG["archivo_entrada"])

    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        raise


if __name__ == "__main__":
    main()
