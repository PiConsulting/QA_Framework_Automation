import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime

# === CONFIGURACIÓN ===
ENDPOINT_BASE = "https://ca-zxkvhdr7ju7ku-orchestrator.graybay-13f3c191.eastus.azurecontainerapps.io/orchestrator"
CONFIG = {
    "archivo_entrada": "preguntas.xlsx",
    "timeout": 120,                # Tiempo máximo por request
    "reintentos": 2,               # Reintentos ante error o timeout
    "delay_entre_requests": 0.5    # Retardo entre preguntas
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


def enviar_pregunta(pregunta: str, reintentos: int = 2, timeout: int = 120):
    """Envía una pregunta al endpoint SSE y devuelve la respuesta completa"""
    payload = {"question": pregunta}
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "X-API-KEY": "servel"
    }

    for intento in range(reintentos + 1):
        inicio = time.time()
        try:
            with requests.post(ENDPOINT_BASE, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
                duracion = round(time.time() - inicio, 2)

                if resp.status_code == 200:
                    # Leer línea por línea
                    respuesta = ""
                    for line in resp.iter_lines(decode_unicode=True):
                        if line:
                            # Filtramos el prefijo 'data: ' si existe
                            if line.startswith("data: "):
                                line = line[6:]
                            respuesta += line + "\n"
                    respuesta = respuesta.strip()

                    return {
                        "respuesta": respuesta,
                        "status": resp.status_code,
                        "duracion": duracion,
                        "error": False
                    }
                else:
                    print(f"⚠️ Error {resp.status_code}: {resp.text[:100]}")
                    return {
                        "respuesta": resp.text[:200],
                        "status": resp.status_code,
                        "duracion": duracion,
                        "error": True
                    }

        except requests.exceptions.Timeout:
            if intento < reintentos:
                print(
                    f"⏱️ Timeout, reintentando {intento + 1}/{reintentos}...")
                time.sleep(2)
                continue
            return {"respuesta": f"Timeout después de {timeout}s", "status": None, "duracion": timeout, "error": True}

        except Exception as e:
            if intento < reintentos:
                print(
                    f"🔄 Error: {e}, reintentando {intento + 1}/{reintentos}...")
                time.sleep(2)
                continue
            return {"respuesta": f"Exception: {e}", "status": None, "duracion": 0, "error": True}


def procesar_preguntas(df: pd.DataFrame):
    """Procesa todas las preguntas con el endpoint SSE"""
    resultados = []
    total = len(df)

    print(f"\n🚀 Iniciando prueba con {total} preguntas")
    print(f"🌐 Endpoint: {ENDPOINT_BASE}\n")

    for i, row in df.iterrows():
        pregunta = str(row["pregunta"]).strip()
        if not pregunta or pregunta.lower() == "nan":
            continue

        print(f"[{i+1}/{total}] 📝 {pregunta[:60]}...", end=" ")

        resultado = enviar_pregunta(
            pregunta, CONFIG["reintentos"], CONFIG["timeout"])
        resultados.append({
            "pregunta": pregunta,
            "respuesta": resultado["respuesta"],
            "status": resultado["status"],
            "duracion_seg": resultado["duracion"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        print("✅" if not resultado["error"]
              else "❌", f"{resultado['duracion']}s")
        time.sleep(CONFIG["delay_entre_requests"])

    return resultados


def guardar_resultados(resultados, archivo_entrada):
    """Guarda los resultados en Excel"""
    if not resultados:
        print("⚠️ No hay resultados para guardar")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = Path(archivo_entrada).stem
    archivo_salida = f"resultados_{nombre_base}_{timestamp}.xlsx"

    df = pd.DataFrame(resultados)
    df.to_excel(archivo_salida, index=False)

    print(f"\n✅ Resultados guardados en: {archivo_salida}")


def main():
    try:
        print(f"\n📂 Cargando archivo: {CONFIG['archivo_entrada']}")
        df = cargar_preguntas(CONFIG["archivo_entrada"])
        print(f"✅ {len(df)} preguntas cargadas")

        resultados = procesar_preguntas(df)
        guardar_resultados(resultados, CONFIG["archivo_entrada"])

    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        raise


if __name__ == "__main__":
    main()
 
