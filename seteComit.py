import os
import re
import time
import json
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# ===================== Configuración OpenAI / Azure =====================
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = "azure"
openai.api_version = os.getenv("OPENAI_API_VERSION")

REPHRASER_MODEL = os.getenv("AZURE_DEPLOYMENT_REPHRASER")
EVALUATOR_MODEL = os.getenv("AZURE_DEPLOYMENT_MODEL")

# ===================== SBERT para similitud =====================
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ===================== Parámetros =====================
N_REPHRASES = 3  # por defecto
SIMILARITY_THRESHOLD = 0.7
CONTENT_SAFETY_MESSAGE = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"

# 🔒 Flag para deshabilitar reformulaciones (leer de .env)
# .env -> DISABLE_REPHRASES=1  (para bloquear rephrases)
DISABLE_REPHRASES = os.getenv("DISABLE_REPHRASES", "0") == "1"

# 🕒 NUEVO: sleep entre preguntas (segundos)
# .env -> SLEEP_BETWEEN_QUESTIONS=10
SLEEP_BETWEEN_QUESTIONS = int(os.getenv("SLEEP_BETWEEN_QUESTIONS", "10"))

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

# ===================== Léxico protegido (cadenas canónicas) =====================
# ATENCIÓN: mantener estas entradas exactamente como las necesitan tus tablas
PROTECTED_TERMS = {
    # MERCADOS
    "CAD REGIONALES", "CAD NACIONALES", "SELF INDEP",
    # GRUPOS
    "CENCO", "LA ANONIMA", "CASINO", "GDN", "DIA", "COOP. OBRERA", "CARREFOUR", "COTO", "LA GALLEGA",
    # CADENAS
    "DISCO", "JUMBO", "CHANGOMAS", "PUNTO MAYORISTA", "CARREFOUR GRANDES",
    "SUPER CHANGOMAS", "CARREFOUR PROXY", "CARREFOUR MEDIANOS", "VEA",
    "HIPER CHANGOMAS", "LIBERTAD", "CARREFOUR PEQUEÑOS", "SPID35", "Self Independientes",
    "LA ANONIMA",
    # CATEGORIAS
    "YOGUR BEBIBLE", "CREMA DE LECHE", "ARROZ", "LECHES FLUIDAS", "LECHES SABORIZADAS",
    "CEREALES", "SNACKS DE ARROZ", "LECHE FERMENTADA",
    # TIPO
    "NATURAL", "YAMANI", "UAT", "No definido", "OTROS", "PARBOLIZADO", "REFRIGERADAS",
    "SACHET", "CARNAROLI", "LF", "DOBLE CAROLINA", "INTEGRAL", "SNACKEO", "SABORIZADO", "UNTABLES",
    # SEGMENTO
    "PARA BATIR", "OTROS SABORES", "TRIGO INFLADO", "FUNCIONAL", "ENTERO", "DULCE DE LECHE",
    "VAINILLA", "OTROS", "CHOCOLATADOS", "RESTO", "DESCREMADA", "CAFES", "GALLETAS", "LIGHT",
    "CHOCOLATE", "COPOS DE ARROZ", "AZUCARADOS", "MIEL", "GRANOLA", "SABORIZADOS", "PARA COCINAR",
    "BANANA", "CHANTIILY", "TOSTADITAS", "PILLOWS", "MAIZ INFLADO", "DESCRE DESLACTOSADO",
    "LECHE FERMENTADA", "DOBLE CREMA", "DESLACTOSADA", "BASICOS", "ARROZ INFLADO", "DURAZNO",
    "TUTUCA", "ENTERA", "DESCREMADO C/LACTOSA", "FRUTILLA", "TOSTADAS", "FRUTALES", "SALUDABLES", "AVENA",
    "SABORES MIXTOS",
    # MARCAS (muestra amplia + claves)
    "LA SERENISIMA", "LA SERENISIMA JUNIOR", "YOGURADE", "LS PROTEIN", "ILOLAY", "CAROGRAN",
    "MR. CHOCO", "MANIERI", "YIN YANG", "BARBARA", "EGRAN", "TOSTEX", "YOGURISIMO", "PRIMOR",
    "CROWIE", "LEIVA", "DON MARCOS", "SCOTTI", "ZOO CARTOON", "FLYNN PAFF", "GALLO",
    "CRIOLLITAS", "MORIXE", "LS CLASICO", "LECHELITA", "DOS HERMANOS", "SAN GIORGIO",
    "VANGUARDIA", "LA PAULINA", "RISKY-DIT", "CRACKINES", "MAXIMO", "WINDY", "SAN IGNACIO",
    "MAIZENA", "ZAFRAN", "KUATI", "OTRAS MARCAS", "TREGAR", "SANCOR", "TOSTI", "MOMY", "CAÑUELAS",
    "NANI", "FROOT LOOPS", "NESFIT", "SER", "DANONINO", "ARROCITAS", "EL AMANECER", "CEREAL MIX",
    "ACTIMEL", "BREVISS", "LA SUIPACHENSE", "MOLTO", "CHOCOLINO", "TODDY", "NIKITOS", "LUCCHETTI",
    "NESQUIK", "RIERA", "ARROZEN", "COSALTA", "NESCAO", "CAPRI SNACKS", "TIA MARUCA", "ARROCIÑO",
    "ANGELITA", "DECECCO", "VITA CEREAL", "MANISUR", "CAPULLITOS EN FLOR", "MOLINOS ALA",
    "PATAGONIA GRAINS", "QUAKER", "MOLINOS RIO DE LA PLATA", "CHOCO KRISPIS", "ZUCARITAS",
    # FABRICANTES (muestra + claves)
    "ESTAB SAN IGNACIO", "OTROS FABRICANTES", "LACTEOS MAMUU", "MAKE IT HAPPEN", "BOGAT",
    "FRIGORIFICO EL AMANECER", "UNILEVER", "LA NUEVA", "MILKAUT", "ELCOR", "ARCOR",
    "MOLINO CAÑUELAS", "BABASAL", "PRIMER PREMIO", "GENERAL CEREALS", "MARCAS PROPIAS",
    "MASTELLONE HNOS", "MANFREY", "DAHI", "CERROS TUCUMANOS", "CONOSUR", "VERONICA", "GANDARA",
    "MOLINOS RIO DE LA PLATA",
    # ÁREAS
    "PERIFERIA", "CAPITAL FEDERAL", "SUR", "CORDOBA", "CUYO", "AUSTRAL", "LIT SUR", "LIT NORTE",
    "NOA", "BS. AS. RESTO", "LITORAL", "ANDINA", "RESTO PCIA BS AS + SUR", "GBA"
}

# --------------------- Utilidades para proteger términos ---------------------
def _boundary_pattern(term: str) -> re.Pattern:
    # Límites tipo "no letra/dígito" para no romper palabras compuestas
    return re.compile(rf"(?<![\wÁÉÍÓÚÜÑáéíóúüñ]){re.escape(term)}(?![\wÁÉÍÓÚÜÑáéíóúüñ])", re.IGNORECASE)

def _terms_in(text: str) -> set:
    text = text or ""
    found = set()
    for t in PROTECTED_TERMS:
        if _boundary_pattern(t).search(text):
            found.add(t)
    return found

def _normalize_to_canonical_case(s: str) -> str:
    out = s or ""
    for t in PROTECTED_TERMS:
        out = _boundary_pattern(t).sub(t, out)
    return out

def _enforce_protected_terms(original: str, candidate: str) -> str:
    """
    Garantiza que todos los términos protegidos presentes en 'original' aparezcan EXACTOS en 'candidate'.
    Si falta alguno => retorna 'original'.
    Si están todos => normaliza casing en 'candidate' y devuelve 'candidate'.
    """
    orig_terms = _terms_in(original)
    if not orig_terms:
        return candidate

    for t in orig_terms:
        if not _boundary_pattern(t).search(candidate or ""):
            return original
    return _normalize_to_canonical_case(candidate or "")

def _normalize_json_protected(json_text: str) -> str:
    """
    Intenta normalizar los términos protegidos dentro de valores string del JSON.
    Si no es JSON válido, normaliza a texto plano.
    """
    try:
        obj = json.loads(json_text)
        def walk(x):
            if isinstance(x, str):
                return _normalize_to_canonical_case(x)
            if isinstance(x, list):
                return [walk(v) for v in x]
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            return x
        obj = walk(obj)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return _normalize_to_canonical_case(json_text)

# ===================== LLM helpers =====================
def rephrase_question(question, previous_rephrasings, prompt_version=1):
    """
    Devuelve una reformulación de `question`.
    - Si DISABLE_REPHRASES=1 => devuelve la original.
    - Si hay términos protegidos en la original, la reformulación debe preservarlos EXACTOS;
      si no lo hace, se cae a la original.
    """
    if DISABLE_REPHRASES:
        return question

    protected_list = sorted(PROTECTED_TERMS)
    hard_rule = (
        "REGLA DURA: No cambies, traduzcas ni modifiques NINGUNA de las siguientes cadenas EXACTAS "
        "(si aparecen en la pregunta). Deben mantenerse idénticas, mismo espaciado y puntuación:\n"
        + ", ".join(protected_list) + "\n"
        "Si la pregunta contiene alguna de ellas, deben figurar tal cual en tu salida."
    )

    prompts = [
        "Sos un asistente de parafraseo para QA. Reformulá manteniendo intención y significado.",
        "Actuá como especialista en parafraseo para evaluar robustez de prompts.",
        "Generá variaciones lingüísticas preservando semántica original.",
        "Reformulá la pregunta cambiando forma pero no el fondo.",
        "Reformulá conservando sentido. Ej: '¿Cómo cocino pasta?' → '¿Cuál es la forma de preparar pasta?'",
    ]
    system_prompt = prompts[max(1, min(5, int(prompt_version))) - 1] + "\n\n" + hard_rule

    response = openai.ChatCompletion.create(
        engine=REPHRASER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    candidate = response["choices"][0]["message"]["content"].strip()
    return _enforce_protected_terms(question, candidate)

def generate_followup_locked(pregunta_original: str, respuesta: str) -> str:
    """
    Genera UNA repregunta breve, pero NO permite alterar términos protegidos
    presentes en la pregunta original. Si los altera, devuelve "".
    """
    if DISABLE_REPHRASES:
        return ""

    hard_rule = (
        "REGLA DURA: No cambies, traduzcas ni modifiques NINGUNA de las siguientes "
        "cadenas EXACTAS (si aparecen en la pregunta). Deben mantenerse idénticas:\n"
        + ", ".join(sorted(PROTECTED_TERMS)) + "\n"
        "Si la pregunta contiene alguna, deben figurar tal cual en tu salida."
    )

    prompt = (
        "Genera UNA sola repregunta breve (máx. 25 palabras) que pida la mínima información faltante "
        "para responder correctamente.\n\n"
        f"{hard_rule}\n\n"
        f"Pregunta original: {pregunta_original or ''}\n"
        f"Respuesta generada: {respuesta or ''}\n\n"
        "Devuelve SOLO la repregunta, sin explicación."
    )

    try:
        out = openai.ChatCompletion.create(
            engine=EVALUATOR_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
        )["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

    orig_terms = _terms_in(pregunta_original)
    if orig_terms:
        for t in orig_terms:
            if not _boundary_pattern(t).search(out or ""):
                return ""
        out = _normalize_to_canonical_case(out)
    return out

def generate_answer(question):
    """
    Devuelve JSON de filtros para que tu backend construya el SQL:
    {
      "mercado": str|null,
      "grupo": str|null,
      "cadena": str|null,
      "categoria": str|null,
      "tipo": str|null,
      "segmento": str|null,
      "marca": str|null,
      "area": str|null,
      "periodo": "YYYY-MM" | ["YYYY-MM", ...] | {"from":"YYYY-MM","to":"YYYY-MM"} | null,
      "metricas": ["SOM","ventas","precio_promedio"] | null,
      "top_n": int|null,
      "comparativo": true|false|null
    }
    Nunca responder con texto libre, solo el JSON.
    """
    start_time = time.time()

    protected_list = ", ".join(sorted(PROTECTED_TERMS))
    system = (
        "Eres un parser de consultas de negocio. Devuelve SOLO un JSON válido con filtros "
        "para consultar tablas internas. No expliques nada. No agregues texto fuera del JSON. "
        "Si un campo no aplica, usa null. Los literales presentes en la consulta que pertenezcan al "
        "siguiente listado deben respetarse EXACTOS (sin traducir ni reformular): " + protected_list + "."
    )
    system += (
        "\nSi la consulta menciona 'top' o 'top 5/10', rellena top_n con ese número; si pide comparación "
        "entre dos momentos, usa una lista en 'periodo' con ambos 'YYYY-MM'."
    )

    user = question

    resp = openai.ChatCompletion.create(
        engine=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
    )
    raw = resp["choices"][0]["message"]["content"].strip()

    normalized = _normalize_json_protected(raw)

    try:
        json.loads(normalized)
    except Exception:
        normalized = json.dumps({
            "mercado": None, "grupo": None, "cadena": None, "categoria": None, "tipo": None,
            "segmento": None, "marca": None, "area": None, "periodo": None,
            "metricas": None, "top_n": None, "comparativo": None,
            "_raw": _normalize_to_canonical_case(raw)
        }, ensure_ascii=False)

    elapsed_time = round(time.time() - start_time, 2)
    return normalized, elapsed_time

# ===================== Similitud =====================
def compute_similarity_cosine(text1, text2):
    emb1 = model_sbert.encode(text1 or "", convert_to_tensor=True)
    emb2 = model_sbert.encode(text2 or "", convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def compute_similarity_llm(pregunta, respuesta1, respuesta2):
    prompt = SIMILARITY_PROMPT.replace("{{pregunta}}", pregunta or "")\
        .replace("{{respuesta1}}", respuesta1 or "")\
        .replace("{{respuesta2}}", respuesta2 or "")
    response = openai.ChatCompletion.create(
        engine=EVALUATOR_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
    )
    try:
        similitud = float(response["choices"][0]["message"]["content"].strip())
    except Exception:
        similitud = 0.0
    return similitud

# ===================== Pipeline principal =====================
def procesar_excel(file_path, similarity_method="cosine"):
    """
    Procesa un archivo Excel con preguntas y genera:
      - Pregunta_reformulada (respetando términos protegidos)
      - Respuesta_obtenida (JSON de filtros)
      - Similitud/Similitud_LLM frente a 'Respuesta_deseada' (si tu ground truth también es JSON)
    Columnas esperadas: ID, Pregunta, Respuesta_deseada, (opcional) Fuente
    """

    if not os.path.exists(file_path):
        print(f"❌ No se encontró el archivo: {file_path}")
        return None

    try:
        xls = pd.ExcelFile(file_path)
        resultados = []

        print(f"📋 Archivo encontrado: {file_path}")
        print(f"📄 Hojas encontradas: {xls.sheet_names}")
        print(f"🔄 Procesando {len(xls.sheet_names)} hoja(s)...\n")

        if DISABLE_REPHRASES:
            print("🔒 DISABLE_REPHRASES=1 → Reformulaciones desactivadas. Se usa la pregunta original y 1 iteración.\n")
        else:
            print("🛡️ Léxico protegido activo: si la reformulación altera nombres críticos, se mantendrá la pregunta original.\n")

        for sheet_idx, sheet_name in enumerate(xls.sheet_names, 1):
            print(f"📊 [{sheet_idx}/{len(xls.sheet_names)}] Procesando hoja: '{sheet_name}'")
            df_input = pd.read_excel(xls, sheet_name=sheet_name)

            required_columns = ["ID", "Pregunta", "Respuesta_deseada"]
            missing_columns = [col for col in required_columns if col not in df_input.columns]
            if missing_columns:
                print(f"⚠️  Faltan columnas en hoja {sheet_name}: {missing_columns}")
                print(f"    Columnas encontradas: {list(df_input.columns)}")
                continue

            df_input["Pregunta"] = df_input["Pregunta"].astype(str)
            df_input["Respuesta_deseada"] = df_input["Respuesta_deseada"].astype(str)

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

                if pregunta in (None, "nan") or respuesta_deseada in (None, "nan"):
                    print(f"    ⚠️  Fila {idx + 2}: falta pregunta o respuesta deseada - SALTANDO")
                    continue

                print(f"    🔍 [{idx+1}/{len(df_input)}] ID {id_pregunta}: {pregunta[:80]}{'...' if len(pregunta) > 80 else ''}")
                previous_rephrasings = []

                n_iters = 1 if DISABLE_REPHRASES else N_REPHRASES

                for i in range(n_iters):
                    try:
                        pregunta_reformulada = rephrase_question(pregunta, previous_rephrasings, prompt_version=1)

                        # Generar respuesta (JSON de filtros)
                        respuesta_obtenida, tiempo = generate_answer(pregunta_reformulada)

                        # Calcular similitudes (opcional si tu ground truth también es JSON)
                        similitud_coseno = compute_similarity_cosine(respuesta_obtenida, respuesta_deseada)
                        similitud_llm = compute_similarity_llm(pregunta, respuesta_obtenida, respuesta_deseada)

                        previous_rephrasings.append(pregunta_reformulada)

                        resultados.append({
                            "ID": id_pregunta,
                            "Fuente": fuente,
                            "Pregunta": pregunta,
                            "Respuesta_deseada": respuesta_deseada,
                            "Pregunta_reformulada": pregunta_reformulada,
                            "Respuesta_obtenida": respuesta_obtenida,  # JSON
                            "Fuente_obtenida": "",  # Completar si el modelo provee fuentes
                            "Similitud": f"{round(similitud_coseno * 100, 1)}%",
                            "Similitud_LLM": f"{round(similitud_llm * 100, 1)}%",
                            "Tiempo": tiempo
                        })

                        print(
                            f"      ✅ Iteración {i+1}: "
                            f"Coseno={similitud_coseno*100:.1f}%, "
                            f"LLM={similitud_llm*100:.1f}%, "
                            f"Tiempo={tiempo}s"
                        )

                    except Exception as e:
                        print(f"      ❌ Error en iteración {i+1}: {str(e)[:120]}...")
                        error_msg = str(e)
                        pregunta_ref = pregunta_reformulada if 'pregunta_reformulada' in locals() else ""

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

                # 🕒 NUEVO: Sleep entre PREGUNTAS (filas)
                if SLEEP_BETWEEN_QUESTIONS > 0:
                    print(f"      ⏳ Esperando {SLEEP_BETWEEN_QUESTIONS}s antes de la próxima pregunta...")
                    time.sleep(SLEEP_BETWEEN_QUESTIONS)

            print(f"    ✅ Hoja '{sheet_name}' completada\n")

        columnas_salida = [
            "ID", "Fuente", "Pregunta", "Respuesta_deseada",
            "Pregunta_reformulada", "Respuesta_obtenida",
            "Fuente_obtenida", "Similitud", "Similitud_LLM", "Tiempo"
        ]

        if not resultados:
            print("❌ No se procesaron resultados. Verifique el formato del archivo.")
            return None

        df_result = pd.DataFrame(resultados, columns=columnas_salida)

        # Guardar reporte
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"reporte_llm_{timestamp}.xlsx"
        df_result.to_excel(output_path, index=False)

        print(f"\n✅ Reporte generado: {output_path}")
        print(f"📊 Total de evaluaciones: {len(df_result)}")

        return df_result

    except Exception as e:
        print(f"❌ Error al procesar el archivo: {e}")
        return None
