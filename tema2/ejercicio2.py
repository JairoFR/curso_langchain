from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# ChatOpenAI es el wrapper de LangChain para los modelos de OpenAI.
# Aquí usamos gpt-4o-mini que es rápido y barato, ideal para clasificar.
llm = ChatOpenAI(model="gpt-4o-mini")

# ─────────────────────────────────────────────
# PASO 1: clasificador
# ─────────────────────────────────────────────
# Recibe: {"pregunta": "¿Cómo hago una lista en Python?"}
# Devuelve: {"tipo": "python", "pregunta": "¿Cómo hago una lista en Python?"}
#
# Le preguntamos al LLM que clasifique en una sola palabra.
# .content      → extrae el texto del objeto AIMessage que devuelve llm.invoke()
# .strip()      → elimina espacios o saltos de línea al inicio/final
# .lower()      → normaliza a minúsculas para que coincida con las keys del dict
#
# IMPORTANTE: siempre pasamos "pregunta" al siguiente paso,
# si no la incluimos aquí se pierde para siempre en la cadena.
clasificar = RunnableLambda(lambda x: {
    "tipo": llm.invoke(
        f"Clasifica esta pregunta en una palabra: "
        f"python, javascript o general. Pregunta: {x['pregunta']}"
    ).content.strip().lower(),
    "pregunta": x["pregunta"]
})

# ─────────────────────────────────────────────
# PASO 2: tabla de rutas (tabla de despacho)
# ─────────────────────────────────────────────
# Cada key es un tipo posible que puede devolver el clasificador.
# Cada value es un RunnableLambda con un system prompt especializado.
# Reciben el dict completo {"tipo": ..., "pregunta": ...}
# y devuelven solo el string con la respuesta (.content).
rutas = {
    "python": RunnableLambda(lambda x: llm.invoke(
        f"Eres experto Python. Responde breve: {x['pregunta']}"
    ).content),
    "javascript": RunnableLambda(lambda x: llm.invoke(
        f"Eres experto JavaScript. Responde breve: {x['pregunta']}"
    ).content),
    "general": RunnableLambda(lambda x: llm.invoke(
        f"Responde esta pregunta de programación: {x['pregunta']}"
    ).content),
}

# ─────────────────────────────────────────────
# PASO 3: router
# ─────────────────────────────────────────────
# rutas.get(x["tipo"], rutas["general"])
#   → busca la key x["tipo"] en el dict
#   → si no existe (ej: el LLM devolvió "rust"), usa "general" como fallback
#   → .invoke(x) ejecuta el RunnableLambda elegido con el mismo input
router = RunnableLambda(
    lambda x: rutas.get(x["tipo"], rutas["general"]).invoke(x)
)

# ─────────────────────────────────────────────
# CADENA FINAL
# ─────────────────────────────────────────────
# El operador | conecta los pasos en secuencia (LCEL).
# Output de clasificar → input de router, automáticamente.
#
# Flujo completo:
# {"pregunta": "¿Cómo hago una lista en Python?"}
#        ↓ clasificar
# {"tipo": "python", "pregunta": "..."}
#        ↓ router → elige rutas["python"]
# "Puedes crear una lista así: mi_lista = [1, 2, 3]"
cadena = clasificar | router

resultado = cadena.invoke({"pregunta": "¿Cómo hago una lista en Python?"})
print(resultado)