from langchain_core.runnables import RunnableLambda

paso1 = RunnableLambda(lambda x: f"Numero {x}")

def duplicar_text(texto):
    return [texto] * 2


paso2 = RunnableLambda(duplicar_text)

cadena =  paso1 | paso2

print(cadena.invoke(1))

