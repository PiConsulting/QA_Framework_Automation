# QA Challenge — Secured (PI + Hallucination + Follow-up)

Pipeline para **QA de LLMs** con:
- Reformulación (N variaciones)
- **Detección de Prompt Injection** (reglas locales + opcional **Azure Prompt Shields**)
- Respuesta con **System seguro**
- Similitudes (SBERT coseno + LLM)
- **Self-check** de alucinación + **repregunta** automática si hay baja confianza
- **Reporte** XLSX con métricas clave

## Requisitos
- Python 3.9+
- Azure OpenAI / OpenAI (usamos `openai==0.28.0` con `engine=` para Azure)
- Paquetes de `requirements.txt`

## Variables de entorno (`.env`)
```env
# OpenAI / Azure OpenAI
OPENAI_API_KEY=...
OPENAI_API_BASE=https://<recurso>.openai.azure.com/
OPENAI_API_TYPE=azure
OPENAI_API_VERSION=2024-02-15-preview

# Deployments (engines) en Azure
AZURE_DEPLOYMENT_REPHRASER=<deployment rephraser>
AZURE_DEPLOYMENT_MODEL=<deployment evaluator>

# Optional: Azure Prompt Shields (Content Safety)
USE_PROMPT_SHIELDS=1
CS_ENDPOINT=https://<recurso-cs>.cognitiveservices.azure.com
CS_KEY=<clave-cs>

# Parámetros (opcionales)
N_REPHRASES=3
SIMILARITY_THRESHOLD=0.7
SELF_CHECK_THRESHOLD=0.65
REQUEST_TIMEOUT_SECONDS=30
