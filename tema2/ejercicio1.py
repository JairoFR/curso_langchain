from langchain_core.runnables import RunnableLambda, RunnableParallel


############################################
#1.- Normalizar texto - Crea una cadena de 3 pasos que: quite espacios extra, convierta a minúsculas y cuente las palabras.
############################################
entrada = "  Hola   Mundo   LangChain  "
# tu cadena aquí
# salida esperada: 3  (número de palabras)

#a- quitar espcio extra
quitar_espacio = RunnableLambda(lambda x: " ".join(x.split()))

#b- convertir a minuscula
a_minuscula = RunnableLambda(lambda x: x.lower())

#c- contar las palabras
contar_palabra = RunnableLambda(lambda x: len(x.split()))

cadena = quitar_espacio | a_minuscula | contar_palabra

#print(cadena.invoke(entrada))

#############################################
#2.- Pipeline matemático - Construye una cadena que reciba un número y aplique: doblar → sumar 10 → elevar al cuadrado. Usa .batch() para probarla con varios inputs a la vez.
############################################
numeros = [1, 2, 3, 4, 5]
# cadena.batch(numeros) → ?

doblar = RunnableLambda(lambda x: x * 2)

sumar_10 = RunnableLambda(lambda x: x + 10)

al_cuadrado = RunnableLambda(lambda x: x**2)

cadena = doblar | sumar_10 | al_cuadrado

#.batch() es útil en la vida real cuando tienes, por ejemplo, 100 textos que procesar con un LLM y no quieres hacerlo uno por uno
#print(cadena.batch(numeros))



############################################
#3.- Transformar un dict - Recibe un dict con nombre y edad. La cadena debe: formatear un saludo, luego verificar si es mayor de edad y agregar ese campo al dict.
############################################
entrada = {"nombre": "Ana", "edad": 17}
# salida esperada:
# {"saludo": "Hola Ana", "mayor_de_edad": False}

saludo = RunnableLambda(lambda x: {
                        "saludo": f"Hola {x["nombre"]}",
                        "edad": x["edad"]
                        })

verificar_edad = RunnableLambda( lambda x: {
                        "saludo": x["saludo"],
                        "mayor_de_edad": x["edad"] >= 18
                        })

cadena = saludo| verificar_edad

#print(cadena.invoke(entrada))

############################################
#4.- Ramificación con RunnableParallel - Combina RunnableLambda con RunnableParallel para procesar un texto y obtener en paralelo: su longitud, sus palabras únicas y si contiene la palabra "python".
############################################

entrada = "python es genial y python es poderoso"
# salida esperada:
# {"longitud": 38, "palabras_unicas": 6, "tiene_python": True}
####### con cadena ###########
# paso previo: limpiar el texto
limpiar = RunnableLambda(lambda x: x.strip().lower())

analisis = RunnableParallel(
    longitud        = RunnableLambda(lambda x: len(x)),
    palabras_unicas = RunnableLambda(lambda x: len(set(x.split()))),
    tiene_python    = RunnableLambda(lambda x: "python" in x)
)

# paso final: resumir resultado
resumir = RunnableLambda(lambda x: 
    f"Texto de {x['longitud']} chars, "
    f"{x['palabras_unicas']} palabras únicas, "
    f"python={'sí' if x['tiene_python'] else 'no'}"
)

cadena = limpiar | analisis | resumir
print("---------------------")
print(cadena.invoke("  Python ES genial y PYTHON es poderoso  "))
# → "Texto de 38 chars, 6 palabras únicas, python=sí"
##########################################################################
####### sin cadena ###########
nalisis = RunnableParallel(
    longitud      = RunnableLambda(lambda x: len(x)),
    palabras_unicas = RunnableLambda(lambda x: len(set(x.split()))),
    tiene_python  = RunnableLambda(lambda x: "python" in x.lower())
)

entrada = "python es genial y python es poderoso"
print("---------------------")
print(analisis.invoke(entrada))
# → {"longitud": 38, "palabras_unicas": 6, "tiene_python": True}

#5.- Cadena con lógica condicional - Crea un router: recibe un dict con tipo ("saludo" o "despedida") y nombre. La cadena debe detectar el tipo y aplicar la transformación correspondiente usando un lambda que bifurca el flujo.

e1 = {"tipo": "saludo", "nombre": "Carlos"}
e2 = {"tipo": "despedida", "nombre": "Carlos"}
# e1 → "¡Hola, Carlos! Bienvenido."
# e2 → "¡Hasta luego, Carlos! Que te vaya bien."


############ PRIMERA FORMA ########
# Paso 1: detectar el tipo y generar el mensaje
router = RunnableLambda(lambda x:
    f"¡Hola, {x['nombre']}! Bienvenido."
    if x["tipo"] == "saludo"
    else f"¡Hasta luego, {x['nombre']}! Que te vaya bien."
)

e1 = {"tipo": "saludo",    "nombre": "Carlos"}
e2 = {"tipo": "despedida", "nombre": "Carlos"}

print(router.invoke(e1))  # → ¡Hola, Carlos! Bienvenido.
print(router.invoke(e2))  # → ¡Hasta luego, Carlos! Que te vaya bien.



############# SEGUNDA FORMA ########

# Cada tipo mapea a su propio RunnableLambda
rutas = {
    "saludo":    RunnableLambda(lambda x: f"¡Hola, {x['nombre']}! Bienvenido."),
    "despedida": RunnableLambda(lambda x: f"¡Hasta luego, {x['nombre']}! Que te vaya bien."),
    "gracias":   RunnableLambda(lambda x: f"¡De nada, {x['nombre']}!"),
}

# El router elige qué rama ejecutar según el tipo
router = RunnableLambda(lambda x: rutas[x["tipo"]].invoke(x))

e1 = {"tipo": "saludo",    "nombre": "Carlos"}
e2 = {"tipo": "despedida", "nombre": "Carlos"}
e3 = {"tipo": "gracias",   "nombre": "Carlos"}

print(router.invoke(e1))  # → ¡Hola, Carlos! Bienvenido.
print(router.invoke(e2))  # → ¡Hasta luego, Carlos! Que te vaya bien.
print(router.invoke(e3))  # → ¡De nada, Carlos!


