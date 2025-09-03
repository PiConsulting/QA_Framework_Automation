# QA Challenge

Este proyecto está diseñado para evaluar y comparar la calidad de respuestas generadas por modelos de lenguaje (LLMs) ante preguntas reformuladas automáticamente. El flujo principal consiste en tomar un archivo Excel con preguntas y respuestas esperadas, reformular las preguntas varias veces, obtener respuestas del modelo, calcular la similitud con la respuesta esperada y generar un reporte con los resultados.

## Estructura del Proyecto

```
qa-challenge/
│
├── archivo.xlsx           # Archivo de entrada con preguntas y respuestas esperadas
├── reporte_llm.xlsx       # Reporte generado con los resultados de la evaluación
├── functions.py           # Funciones principales para procesamiento, reformulación y evaluación
├── main.py                # Script principal para ejecutar el procesamiento
├── requirements.txt       # Dependencias del proyecto
└── venv/                  # Entorno virtual de Python
```

## Requisitos

- Python 3.8 o superior
- Cuenta y credenciales de OpenAI o Azure OpenAI (si se usan modelos de la API)
- Paquetes listados en `requirements.txt`

## Instalación

1. **Clona el repositorio y entra a la carpeta:**
   ```bash
   git clone <url-del-repo>
   cd qa-challenge
   ```

2. **Crea y activa un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura tus variables de entorno**  
   Crea un archivo `.env` en la raíz del proyecto con tus claves y endpoints de OpenAI/Azure OpenAI, por ejemplo:
   ```
   OPENAI_API_KEY=tu_clave
   OPENAI_API_BASE=https://<tu-recurso>.openai.azure.com/
   OPENAI_DEPLOYMENT_MODEL=modelo-del-deployment-opeanai
   OPENAI_API_VERSION=2023-05-15
   AZURE_DEPLOYMENT_MODEL=modelo-del-deployment-de-azure
   AZURE_DEPLOYMENT_REPHRASER=nombre-del-deployment

   ```

## Uso

1. **Prepara el archivo de entrada**  
   Asegúrate de que `archivo.xlsx` tenga al menos tres columnas:  
   - `Pregunta`
   - `Fuente (si aplica)`
   - `Respuesta deseada`

2. **Ejecuta el procesamiento principal:**
   ```bash
   python main.py
   ```
   Esto generará el archivo `reporte_llm.xlsx` con los resultados.

3. **(Opcional) Ejecuta pruebas:**
   ```bash
   python test.py
   ```

## ¿Qué hace el reporte?

El archivo `reporte_llm.xlsx` contendrá, para cada pregunta:
- La pregunta original
- Reformulaciones generadas
- Respuestas del modelo
- Similitud con la respuesta esperada
- Métricas agregadas

## Notas

- Si usas Azure OpenAI, asegúrate de que el nombre del deployment y el endpoint sean correctos.
- Si ves errores como `Resource not found`, revisa las variables de entorno y el nombre del deployment/modelo.

## Evaluación/Clasificación
- El porcentaje de aceptación para un POC es del 70% y para un deploy un píso del 80%

## Licencia

MIT 
