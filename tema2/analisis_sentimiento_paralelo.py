from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel  # RunnableParallel ejecuta ramas en paralelo
from langchain_openai import ChatOpenAI
import json

load_dotenv()

############################################################
# RunnableParallel es ideal cada vez que tenés dos o más operaciones independientes 
# sobre el mismo input — el caso clásico en LangChain es exactamente este: resumir + clasificar al mismo tiempo.
############################################################


# ── Configuración del modelo ──────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Paso 1: Preprocesamiento ──────────────────────────────────────────────────
def preprocess_text(text):
    """Limpia el texto: elimina espacios extras y limita a 500 caracteres."""
    return text.strip()[:500]

preprocessor = RunnableLambda(preprocess_text)


# ── Paso 2 (paralelo): Resumen y Sentimiento ──────────────────────────────────
def generate_summary(text):
    """
    Rama A del paralelo.
    Recibe el texto ya preprocesado y devuelve un resumen de una oración.
    """
    prompt = f"Resume en una sola oracion: {text}"
    response = llm.invoke(prompt)
    return response.content          # AIMessage → str


def analyze_sentiment(text):
    """
    Rama B del paralelo.
    Recibe el mismo texto preprocesado y devuelve un dict con sentimiento y razón.
    Ambas ramas (A y B) reciben el MISMO input y corren al mismo tiempo.
    """
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde unicamente en formato JSON Valido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificacion breve"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "error en analisis"}


# RunnableParallel recibe un dict con nombre → Runnable
# Cuando se invoca con un texto, lanza AMBAS ramas simultáneamente
# y devuelve: {"resumen": <resultado_A>, "sentimiento_data": <resultado_B>}
parallel_step = RunnableParallel(
    resumen          = RunnableLambda(generate_summary),   # ─┐ corren
    sentimiento_data = RunnableLambda(analyze_sentiment)   # ─┘ en paralelo
)


# ── Paso 3: Merge de resultados ───────────────────────────────────────────────
def merge_results(data):
    """
    Recibe el dict que produce RunnableParallel:
      {
        "resumen":          str,
        "sentimiento_data": {"sentimiento": str, "razon": str}
      }
    Y lo aplana en una estructura final limpia.
    """
    return {
        "resumen":     data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon":       data["sentimiento_data"]["razon"]
    }

merger = RunnableLambda(merge_results)


# ── Cadena completa ───────────────────────────────────────────────────────────
#
#   texto_crudo
#       │
#   preprocessor          ← limpia el texto
#       │
#   parallel_step         ← divide en DOS ramas simultáneas
#     ├── generate_summary      → "resumen"
#     └── analyze_sentiment     → "sentimiento_data"
#       │
#   merger                ← une los resultados en un dict final
#
chain = preprocessor | parallel_step | merger


# ── Prueba ────────────────────────────────────────────────────────────────────
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde."
]

#for texto in textos_prueba:
#    resultado = chain.invoke(texto)
#    print(f"Texto:       {texto}")
#    print(f"Resumen:     {resultado['resumen']}")
#    print(f"Sentimiento: {resultado['sentimiento']} — {resultado['razon']}")
#    print("-" * 50)


resultados_batch = chain.batch(textos_prueba)
print(resultados_batch)


