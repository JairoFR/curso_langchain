from dotenv import load_dotenv           # Carga variables de entorno desde .env
from langchain_openai import ChatOpenAI  # Cliente para modelos de OpenAI
from langchain_core.prompts import PromptTemplate    # Plantilla de prompt con variables
from langchain_classic.chains import LLMChain        # Cadena que une prompt + modelo

load_dotenv()  # Lee el .env y carga OPENAI_API_KEY al entorno

# Inicializa el modelo GPT-4o-mini con creatividad media (0.7)
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define la plantilla del prompt con la variable {nombre}
plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre.\nNombre del usuario: {nombre}\nAsistente:"
)

# Une el prompt y el modelo en una cadena ejecutable
chain = plantilla | chat

# Ejecuta la cadena pasando el valor de {nombre}
resultado = chain.invoke({"nombre": "Carlos"})
print(resultado.content)


