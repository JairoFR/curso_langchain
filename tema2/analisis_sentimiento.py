from dotenv import load_dotenv           # Carga variables de entorno desde .env
from langchain_core.runnables import RunnableLambda  # Convierte funciones Python en "Runnables" compatibles con LangChain
from langchain_openai import ChatOpenAI  # Wrapper de LangChain para los modelos de OpenAI
import json

load_dotenv()  # Lee el archivo .env y expone OPENAI_API_KEY como variable de entorno

# ── Configuración del modelo ──────────────────────────────────────────────────
# temperature=0 → respuestas deterministas (sin creatividad/aleatoriedad)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


#########################################################
# RunnableLambda es el "adaptador" que le dice a LangChain "esta función Python normal también sabe 
# responder a .invoke(), .stream() y .batch()". Sin él no podrías usar | para encadenarla.
#El operador | construye un RunnableSequence en tiempo de definición; no ejecuta nada hasta que llamas .invoke(). 
# Piénsalo como armar un pipeline de tuberías antes de abrir la llave.
#########################################################

# ── Paso 1: Preprocesamiento ──────────────────────────────────────────────────
def preporcess_text(text):
    """
    Limpia el texto eliminando espacios extras y limitando la longitud.
    Es el primer eslabón de la cadena; garantiza que el input sea uniforme
    antes de enviarlo al modelo.
    """
    text = text.strip()[:500]  # strip() quita espacios/saltos al inicio y fin; [:500] limita a 500 caracteres
    return text

# RunnableLambda envuelve la función para que pueda usarse dentro de una chain
# con el operador pipe  |  (igual que los demás componentes de LangChain)
preprocessor = RunnableLambda(preporcess_text)


# ── Paso 2a: Generación de resumen ───────────────────────────────────────────
def generate_summary(text):
    """
    Genera un resumen de una sola oración.
    llm.invoke(prompt) envía el texto al modelo y devuelve un AIMessage;
    .content extrae solo el string de la respuesta.
    """
    prompt = f"Resume en una sola oracion: {text}"
    response = llm.invoke(prompt)   # Llamada síncrona al API de OpenAI
    return response.content         # AIMessage → str


# ── Paso 2b: Análisis de sentimiento ─────────────────────────────────────────
def analyze_sentiment(text):
    """
    Analiza el sentimiento y devuelve un dict estructurado.
    Se le pide al modelo responder *solo* en JSON para poder parsearlo
    de forma confiable; el try/except captura casos donde el modelo
    no respete el formato pedido.
    """
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde unicamente en formato JSON Valido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificacion breve"}}

        Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)   # Convierte el string JSON → dict Python
    except json.JSONDecodeError:
        # Fallback: si el modelo devuelve texto con mal formato no rompemos el flujo
        return {"sentimiento": "neutro", "razon": "error en analisis"}


# ── Paso 3: Merge de resultados ───────────────────────────────────────────────
def merge_results(data):
    """
    Aplana el dict intermedio en una estructura final limpia.
    Recibe: {"resumen": str, "sentimiento_data": {"sentimiento": str, "razon": str}}
    Devuelve: {"resumen": str, "sentimiento": str, "razon": str}
    """
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon":       data["sentimiento_data"]["razon"]
    }


# ── Orquestador: ejecuta ambas ramas y une resultados ────────────────────────
def process_one(t):
    """
    Función central que:
      1. Llama a generate_summary  → obtiene resumen
      2. Llama a analyze_sentiment → obtiene dict de sentimiento
      3. Llama a merge_results     → unifica ambos en un solo dict de salida
    Nota: aquí las dos llamadas al LLM son secuenciales. Para paralelizarlas
    en LangChain se usaría RunnableParallel en lugar de llamadas manuales.
    """
    resumen          = generate_summary(t)
    sentimiento_data = analyze_sentiment(t)
    return merge_results({
        "resumen":          resumen,
        "sentimiento_data": sentimiento_data
    })

# Convierte process_one en Runnable para poder encadenarlo con |
process = RunnableLambda(process_one)


# ── Textos de prueba ──────────────────────────────────────────────────────────
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

# ── Cadena completa ───────────────────────────────────────────────────────────
# El operador | conecta Runnables en secuencia (igual que Unix pipes):
#   texto_crudo → preprocessor → process → resultado_final
chain = preprocessor | process

for texto in textos_prueba:
    resultado = chain.invoke(texto)   # Ejecuta toda la cadena con un solo texto
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)