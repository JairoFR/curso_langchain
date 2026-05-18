# --- Importaciones ---
from langchain_openai import ChatOpenAI                        # Cliente para conectarse a modelos de OpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # Tipos de mensajes del chat
import streamlit as st                                         # Framework para construir la interfaz web
from dotenv import load_dotenv                                 # Lee variables de entorno desde archivo .env
from langchain_core.prompts import PromptTemplate              # Plantilla reutilizable para construir prompts

# --- Carga de variables de entorno ---
# Lee el archivo .env y pone OPENAI_API_KEY disponible en el sistema
load_dotenv()

# --- Configuración de la página ---
# Define el título de la pestaña del navegador y el ícono
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")

# Título principal visible en la app
st.title("Chatbot Básico con Langchain")

# Subtítulo descriptivo en formato markdown
st.markdown(
    "Este es un *chatbot de ejemplo* construido con **Langchain + Streamlit**. "
    "Escribe tu mensaje abajo para usar el bot."
)

# --- Barra lateral de configuración ---
with st.sidebar:
    st.header("Configuración")

    # Slider para controlar la creatividad del modelo (0=determinista, 1=muy creativo)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)

    # Selectbox para elegir qué modelo de OpenAI usar
    model_name = st.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"])

    # Se recrea el modelo en cada rerun para reflejar los parámetros actuales del sidebar
    chat_model = ChatOpenAI(model=model_name, temperature=temperature)

  
# --- Plantilla del prompt ---
# Define la personalidad del bot y el formato que recibirá el modelo.
# LangChain infiere automáticamente las variables {historial} y {mensaje} del template.
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template= """Eres un asistente útil y amigable llamado ChatBot Pro.

Historial de conversación:
{historial}

Responde de manera clara y concisa a la siguiente pregunta: {mensaje} """
)

# --- Cadena LCEL ---
# El operador | conecta el output del PromptTemplate al input del ChatModel.
# Equivale a: formatear el prompt → enviarlo al modelo → recibir respuesta
cadena = prompt_template | chat_model

# --- Inicialización del historial ---
# session_state persiste datos entre reruns de Streamlit.
# Solo inicializa la lista la primera vez que carga la app.
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# --- Renderizado del historial en pantalla ---
# Recorre todos los mensajes guardados y los muestra en la interfaz
for msg in st.session_state.mensajes:
    # Los SystemMessage son instrucciones internas; no se muestran al usuario
    if isinstance(msg, SystemMessage):
        continue

    # Determina el rol para el estilo visual del globo de chat
    role = "assistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(msg.content)  # Renderiza el contenido del mensaje en markdown

# Botón para limpiar el historial y empezar una conversación nueva
if st.button("🗑️ Nueva conversación"):
    st.session_state.mensajes = []  # Vacía la lista de mensajes
    st.rerun()                      # Fuerza a Streamlit a re-ejecutar el script limpiando la pantalla


# --- Entrada del usuario ---
# Muestra la barra de texto fija en la parte inferior. Devuelve el texto o None si está vacío.
pregunta = st.chat_input("Escribe tu mensaje:")

if pregunta:
    # Muestra el mensaje del usuario inmediatamente sin esperar la respuesta del modelo
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Construye el historial como texto plano para pasarlo al prompt
    # Convierte cada objeto mensaje en "Usuario: ..." o "Asistente: ..."
    historial_texto = "\n".join([
        f"Usuario: {m.content}" if isinstance(m, HumanMessage) else f"Asistente: {m.content}"
        for m in st.session_state.mensajes
    ])

    try:
        with st.chat_message("assistant"):
            # st.empty() crea un contenedor vacío que se puede actualizar en cada chunk
            response_placeholder = st.empty()
            full_response = ""

            # cadena.stream() envía el prompt y recibe la respuesta en fragmentos (chunks)
            # en lugar de esperar a que termine toda la respuesta
            for chunk in cadena.stream({
                "mensaje": pregunta,
                "historial": historial_texto if historial_texto else "Sin historial aún."
            }):
                # str() evita error de tipos ya que chunk.content puede ser str o list
                full_response += str(chunk.content)

                # Actualiza el placeholder con el texto acumulado + cursor parpadeante
                response_placeholder.markdown(full_response + "▌")

            # Al terminar el streaming, reemplaza el contenido sin el cursor
            response_placeholder.markdown(full_response)

        # Guarda el mensaje del usuario en el historial DESPUÉS del streaming
        st.session_state.mensajes.append(HumanMessage(content=pregunta))

        # Guarda la respuesta completa del asistente en el historial
        st.session_state.mensajes.append(AIMessage(content=full_response))

    except Exception as e:
        # Muestra el error en pantalla si algo falla durante la generación
        st.error(f"Error al generar respuesta: {str(e)}")

        # Sugiere al usuario verificar la API key como causa más común de error
        st.info("Verifica que tu API Key de OpenAI esté configurada correctamente.")