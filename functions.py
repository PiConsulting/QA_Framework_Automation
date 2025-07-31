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
SIMILARITY_THRESHOLD = 0.8
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

def rephrase_question(question, previous_rephrasings):
    response = openai.ChatCompletion.create(
        engine=REPHRASER_MODEL,
        messages=[
            {"role": "system", "content": "Reformulá esta pregunta para que mantenga su significado. No uses las siguientes reformulaciones: {}".format(previous_rephrasings)},
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
    # Leer todas las hojas del archivo Excel
    xls = pd.ExcelFile(file_path)
    resultados = []

    for sheet_name in xls.sheet_names:
        print(f"Procesando hoja: {sheet_name}")
        df_input = pd.read_excel(xls, sheet_name=sheet_name)
        df_input["Respuesta esperada"] = df_input["Respuesta esperada"].astype(str)

        for idx, row in df_input.iterrows():
            print(f"Procesando fila {idx+1}: {row['Pregunta']}")
            pregunta = row.get("Pregunta")
            esperada = row.get("Respuesta esperada")

            if pd.isna(pregunta) or pd.isna(esperada):
                print(f"⚠️  Hoja {sheet_name} - Fila {idx + 2}: falta pregunta o respuesta esperada")
                continue

            similitudes = []
            previous_rephrasings = []
            print("--------------------------------")

            for i in range(N_REPHRASES):
                try:
                    pregunta_reformulada = rephrase_question(pregunta, previous_rephrasings)
                    respuesta, tiempo = generate_answer(pregunta_reformulada)
                    similitud_coseno = compute_similarity_cosine(respuesta, esperada)
                    similitud_llm = compute_similarity_llm(pregunta, respuesta, esperada)

                    similitudes.append(similitud_coseno)  # Puedes elegir cuál usar para promedios
                    previous_rephrasings.append(pregunta_reformulada)
                    resultados.append({
                        "Hoja": sheet_name,
                        "Fila": idx + 2,
                        "Pregunta original": pregunta,
                        "Pregunta reformulada": pregunta_reformulada,
                        "Respuesta generada": respuesta,
                        "Respuesta esperada": esperada,
                        "Similitud Coseno": round(similitud_coseno, 4),
                        "Similitud LLM": round(similitud_llm, 4),
                        "Tiempo (s)": tiempo,
                        ">0.8 Coseno": similitud_coseno > SIMILARITY_THRESHOLD,
                        ">0.8 LLM": similitud_llm > SIMILARITY_THRESHOLD
                    })

                except Exception as e:
                    print(f"❌ Error en hoja {sheet_name} - fila {idx + 2}: {e}")
                    error_msg = str(e)
                    if CONTENT_SAFETY_MESSAGE in error_msg:
                        resultados.append({
                            "Hoja": sheet_name,
                            "Fila": idx + 2,
                            "Pregunta original": pregunta,
                            "Pregunta reformulada": pregunta_reformulada if 'pregunta_reformulada' in locals() else "",
                            "Respuesta generada": "[Content Safety Triggered]",
                            "Respuesta esperada": esperada,
                            "Similitud": 0.0,
                            "Tiempo (s)": "",
                            ">0.8": False
                        })

            # Métricas por pregunta original
            if similitudes:
                similitud_coseno_prom = round(np.mean([r["Similitud Coseno"] for r in resultados if r["Pregunta original"] == pregunta and r["Hoja"] == sheet_name]), 4)
                similitud_llm_prom = round(np.mean([r["Similitud LLM"] for r in resultados if r["Pregunta original"] == pregunta and r["Hoja"] == sheet_name]), 4)
                resultados.append({
                    "Hoja": sheet_name,
                    "Fila": idx + 2,
                    "Pregunta original": pregunta,
                    "Pregunta reformulada": "- PROMEDIO -",
                    "Respuesta generada": "",
                    "Respuesta esperada": esperada,
                    "Similitud Coseno": similitud_coseno_prom,
                    "Similitud LLM": similitud_llm_prom,
                    "Tiempo (s)": "",
                    ">0.8 Coseno": similitud_coseno_prom > SIMILARITY_THRESHOLD,
                    ">0.8 LLM": similitud_llm_prom > SIMILARITY_THRESHOLD
                })

    df_result = pd.DataFrame(resultados)
    output_path = "reporte_llm.xlsx"
    df_result.to_excel(output_path, index=False)
    print(f"✅ Reporte generado: {output_path}")