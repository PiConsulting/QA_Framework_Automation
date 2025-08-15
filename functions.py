import os
import time
import openai
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# Configuración de OpenAI / Azure AI Foundry
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = "azure"
openai.api_version = os.getenv("OPENAI_API_VERSION")

REPHRASER_MODEL = os.getenv("AZURE_DEPLOYMENT_REPHRASER")
EVALUATOR_MODEL = os.getenv("AZURE_DEPLOYMENT_MODEL")

# SBERT para similitud
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Parámetros
N_REPHRASES = 3  # Cantidad de reformulaciones
SIMILARITY_THRESHOLD = 0.7
CONTENT_SAFETY_MESSAGE = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"

SIMILARITY_PROMPT = (
    "Compara la similitud de las siguientes dos respuestas a esta pregunta\n"
    "{{pregunta}}\n"
    "Respuesta 1:\n"
    "{{respuesta1}}\n"
    "Respuesta 2:\n"
    "{{respuesta2}}\n"
    "\n"
    "**El número debe variar entre 0 y 1, siendo 0 respuestas completamente distintas y 1 respuestas idénticas**\n"
    "**Solo responde con el número**"
)

def rephrase_question(question, previous_rephrasings, prompt_version=1):
    prompts = [
        # Opción 1: Más directa y colaborativa
        (
            "Sos un asistente especializado en reformular preguntas para testing de sistemas de IA. "
            "Tu trabajo es reformular la pregunta manteniendo exactamente el mismo significado e intención, "
            "pero usando diferentes palabras y estructura. Esto es parte de un proceso de evaluación "
            "para mejorar la seguridad y robustez de contenido. "
            "Reformulaciones ya usadas que NO debes repetir: {}".format(previous_rephrasings)
        ),
        # Opción 2: Más técnica y profesional
        (
            "Actúa como un especialista en parafraseo para testing de modelos de lenguaje. "
            "Reformulá la siguiente pregunta preservando su significado semántico completo, "
            "pero alterando la sintaxis y vocabulario. Esta reformulación es necesaria para "
            "evaluaciones de consistencia y mejoras en filtros de seguridad de contenido. "
            "Evitá estas reformulaciones previas: {}".format(previous_rephrasings)
        ),
        # Opción 3: Enfoque en la utilidad del proceso
        (
            "Tu función es generar variaciones lingüísticas de preguntas para testing automatizado. "
            "Reformulá la pregunta manteniendo su significado original pero cambiando su formulación. "
            "Esto permite evaluar la consistencia de respuestas y optimizar sistemas de moderación. "
            "No reutilices estas reformulaciones: {}".format(previous_rephrasings)
        ),
        # Opción 4: Más casual pero efectiva
        (
            "Necesito que reformules esta pregunta de manera diferente pero manteniendo el mismo sentido. "
            "Es para testear cómo responde el sistema a diferentes formas de hacer la misma pregunta "
            "y así mejorar los controles de seguridad. Simplemente cambiá las palabras y la estructura. "
            "Ya se usaron estas versiones: {}".format(previous_rephrasings)
        ),
        # Opción 5: Con ejemplo
        (
            "Reformulá esta pregunta manteniendo exactamente el mismo significado. "
            "Ejemplo: '¿Cómo cocino pasta?' → '¿Cuál es la forma de preparar pasta?' "
            "Esto es para testing de consistencia en sistemas de IA y mejora de content safety. "
            "No uses estas reformulaciones previas: {}".format(previous_rephrasings)
        )
    ]
    # prompt_version va de 1 a 5
    system_prompt = prompts[prompt_version - 1]

    response = openai.ChatCompletion.create(
        engine=REPHRASER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_answer(question):
    start_time = time.time()
    response = openai.ChatCompletion.create(
        engine=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": "Respondé la siguiente pregunta de forma precisa."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    elapsed_time = round(time.time() - start_time, 2)
    return answer, elapsed_time


def compute_similarity_cosine(text1, text2):
    emb1 = model_sbert.encode(text1, convert_to_tensor=True)
    emb2 = model_sbert.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def compute_similarity_llm(pregunta, respuesta1, respuesta2):
    prompt = SIMILARITY_PROMPT.replace("{{pregunta}}", pregunta)\
        .replace("{{respuesta1}}", respuesta1)\
        .replace("{{respuesta2}}", respuesta2)
    response = openai.ChatCompletion.create(
        engine=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.0,
    )
    # Intentar extraer el número de la respuesta
    try:
        similitud = float(response["choices"][0]["message"]["content"].strip())
    except Exception:
        similitud = 0.0
    return similitud

def procesar_excel(file_path, similarity_method="cosine"):
    """
    Procesa un archivo Excel con preguntas y evalúa respuestas mediante reformulaciones.
    
    Columnas de entrada esperadas:
    - ID: Identificador único de la pregunta
    - Pregunta: Pregunta original
    - Fuente: Fuente de la pregunta (opcional)
    - Respuesta_deseada: Respuesta esperada para comparación
    
    Columnas de salida:
    - ID, Fuente, Pregunta, Respuesta_deseada, Pregunta_reformulada, 
      Respuesta_obtenida, Fuente_obtenida, Similitud, Similitud_LLM, Tiempo
    """
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"❌ No se encontró el archivo: {file_path}")
        return None
    
    try:
        # Leer todas las hojas del archivo Excel
        xls = pd.ExcelFile(file_path)
        resultados = []
        
        print(f"📋 Archivo encontrado: {file_path}")
        print(f"📄 Hojas encontradas: {xls.sheet_names}")
        print(f"🔄 Procesando {len(xls.sheet_names)} hoja(s)...\n")

        for sheet_idx, sheet_name in enumerate(xls.sheet_names, 1):
            print(f"📊 [{sheet_idx}/{len(xls.sheet_names)}] Procesando hoja: '{sheet_name}'")
            df_input = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Asegurarse de que las columnas necesarias existen
            required_columns = ["ID", "Pregunta", "Respuesta_deseada"]
            missing_columns = [col for col in required_columns if col not in df_input.columns]
            
            if missing_columns:
                print(f"⚠️  Faltan columnas en hoja {sheet_name}: {missing_columns}")
                print(f"    Columnas encontradas: {list(df_input.columns)}")
                continue
            
            # Asegurar que las columnas sean strings
            df_input["Pregunta"] = df_input["Pregunta"].astype(str)
            df_input["Respuesta_deseada"] = df_input["Respuesta_deseada"].astype(str)
            
            # La columna Fuente es opcional
            if "Fuente" not in df_input.columns:
                df_input["Fuente"] = ""
                print(f"    ℹ️  Columna 'Fuente' no encontrada, se usará string vacío")
            else:
                df_input["Fuente"] = df_input["Fuente"].fillna("").astype(str)
            
            print(f"    📝 Filas a procesar: {len(df_input)}")

            for idx, row in df_input.iterrows():
                id_pregunta = row.get("ID")
                pregunta = row.get("Pregunta")
                fuente = row.get("Fuente", "")
                respuesta_deseada = row.get("Respuesta_deseada")

                if pd.isna(pregunta) or pd.isna(respuesta_deseada) or pregunta == "nan" or respuesta_deseada == "nan":
                    print(f"    ⚠️  Fila {idx + 2}: falta pregunta o respuesta deseada - SALTANDO")
                    continue

                print(f"    🔍 [{idx+1}/{len(df_input)}] ID {id_pregunta}: {pregunta[:50]}{'...' if len(pregunta) > 50 else ''}")
                previous_rephrasings = []

                for i in range(N_REPHRASES):
                    try:
                        # Reformular la pregunta
                        pregunta_reformulada = rephrase_question(pregunta, previous_rephrasings, prompt_version=1)
                        
                        # Generar respuesta
                        respuesta_obtenida, tiempo = generate_answer(pregunta_reformulada)
                        
                        # Calcular similitudes
                        similitud_coseno = compute_similarity_cosine(respuesta_obtenida, respuesta_deseada)
                        similitud_llm = compute_similarity_llm(pregunta, respuesta_obtenida, respuesta_deseada)
                        
                        previous_rephrasings.append(pregunta_reformulada)
                        
                        # Agregar resultado con el formato solicitado (porcentajes)
                        resultados.append({
                            "ID": id_pregunta,
                            "Fuente": fuente,
                            "Pregunta": pregunta,
                            "Respuesta_deseada": respuesta_deseada,
                            "Pregunta_reformulada": pregunta_reformulada,
                            "Respuesta_obtenida": respuesta_obtenida,
                            "Fuente_obtenida": "",  # Se puede llenar si el modelo devuelve fuentes
                            "Similitud": f"{round(similitud_coseno * 100, 1)}%",
                            "Similitud_LLM": f"{round(similitud_llm * 100, 1)}%",
                            "Tiempo": tiempo
                        })

                        print(f"      ✅ Reformulación {i+1}: Coseno={similitud_coseno*100:.1f}%, LLM={similitud_llm*100:.1f}%, Tiempo={tiempo}s")

                    except Exception as e:
                        print(f"      ❌ Error en reformulación {i+1}: {str(e)[:50]}...")
                        error_msg = str(e)
                        
                        # Obtener la pregunta reformulada si está disponible
                        pregunta_ref = pregunta_reformulada if 'pregunta_reformulada' in locals() else ""
                        
                        # Manejar errores de content safety
                        if CONTENT_SAFETY_MESSAGE in error_msg:
                            respuesta_error = "[Content Safety Triggered]"
                        else:
                            respuesta_error = f"[Error: {str(e)[:100]}]"
                        
                        resultados.append({
                            "ID": id_pregunta,
                            "Fuente": fuente,
                            "Pregunta": pregunta,
                            "Respuesta_deseada": respuesta_deseada,
                            "Pregunta_reformulada": pregunta_ref,
                            "Respuesta_obtenida": respuesta_error,
                            "Fuente_obtenida": "",
                            "Similitud": "0.0%",
                            "Similitud_LLM": "0.0%",
                            "Tiempo": ""
                        })
            
            print(f"    ✅ Hoja '{sheet_name}' completada\n")

        # Crear DataFrame con las columnas en el orden especificado
        columnas_salida = [
            "ID",
            "Fuente", 
            "Pregunta",
            "Respuesta_deseada",
            "Pregunta_reformulada",
            "Respuesta_obtenida",
            "Fuente_obtenida",
            "Similitud",
            "Similitud_LLM",
            "Tiempo"
        ]
        
        if not resultados:
            print("❌ No se procesaron resultados. Verifique el formato del archivo.")
            return None
        
        df_result = pd.DataFrame(resultados, columns=columnas_salida)
        
        # Generar reporte con timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"reporte_llm_{timestamp}.xlsx"
        df_result.to_excel(output_path, index=False)
        
        # Mostrar estadísticas
        total_evaluaciones = len(df_result)
        
        # Convertir porcentajes de vuelta a números para estadísticas
        similitudes_coseno_num = []
        similitudes_llm_num = []
        
        for _, row in df_result.iterrows():
            # Extraer números de los porcentajes
            try:
                sim_coseno = float(row["Similitud"].replace("%", "")) / 100
                sim_llm = float(row["Similitud_LLM"].replace("%", "")) / 100
                
                if sim_coseno > 0:  # Solo contar evaluaciones exitosas
                    similitudes_coseno_num.append(sim_coseno)
                if sim_llm > 0:
                    similitudes_llm_num.append(sim_llm)
            except:
                continue
        
        evaluaciones_exitosas = len(similitudes_coseno_num)
        
        if evaluaciones_exitosas > 0:
            similitud_coseno_promedio = sum(similitudes_coseno_num) / len(similitudes_coseno_num)
            similitud_llm_promedio = sum(similitudes_llm_num) / len(similitudes_llm_num) if similitudes_llm_num else 0
            
            print(f"\n✅ Reporte generado: {output_path}")
            print(f"📊 Total de evaluaciones: {total_evaluaciones}")
            print(f"✅ Evaluaciones exitosas: {evaluaciones_exitosas}")
            print(f"❌ Evaluaciones con error: {total_evaluaciones - evaluaciones_exitosas}")
            print(f"📈 Similitud Coseno promedio: {similitud_coseno_promedio * 100:.1f}%")
            print(f"📈 Similitud LLM promedio: {similitud_llm_promedio * 100:.1f}%")
            
            # Estadísticas adicionales
            similitudes_altas_coseno = len([s for s in similitudes_coseno_num if s >= SIMILARITY_THRESHOLD])
            similitudes_altas_llm = len([s for s in similitudes_llm_num if s >= SIMILARITY_THRESHOLD])
            
            print(f"🎯 Similitud Coseno >= {SIMILARITY_THRESHOLD*100:.0f}%: {similitudes_altas_coseno}/{evaluaciones_exitosas} ({100*similitudes_altas_coseno/evaluaciones_exitosas:.1f}%)")
            print(f"🎯 Similitud LLM >= {SIMILARITY_THRESHOLD*100:.0f}%: {similitudes_altas_llm}/{evaluaciones_exitosas} ({100*similitudes_altas_llm/evaluaciones_exitosas:.1f}%)")
        else:
            print(f"\n⚠️  Reporte generado: {output_path}")
            print(f"❌ No se completaron evaluaciones exitosas")
        
        return df_result
        
    except Exception as e:
        print(f"❌ Error al procesar el archivo: {e}")
        return None