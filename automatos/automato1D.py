import numpy as np
import matplotlib.pyplot as plt

# converte um número inteiro para sua representação binária (0—255)
def converte_binario(numero):
    binario = bin(numero)
    binario = binario[2:]
    if len(binario) < 8:
        zeros = [0] * (8-len(binario))
    binario = zeros + list(binario)
    return list (binario)

# Início do script
MAX = 500
g = np.zeros(1000)
ng = np.zeros(1000)
regra = int(input ( 'Entre com o número da regra: ' ) )
codigo = converte_binario(regra)

# Matriz em que cada linha armazena uma geraçao do autômato
matriz_evolucao = np.zeros((MAX, len(g)))

# Define geração inicial
g[len(g)//2]=1

# loop principal: atualiza as gerações
for i in range(MAX):
    matriz_evolucao[i,:] = g
    # percorrendo células da geração atual
    for j in range(len(g)):
        if   (g[j-1] == 0 and g[j] == 0 and g[(j+1)%len(g)] == 0):
            ng[j] = int(codigo[7])
        elif (g[j-1] == 0 and g[j] == 0 and g[(j+1)%len(g)] == 1):
            ng[j] = int(codigo[6])
        elif (g[j-1] == 0 and g[j] == 1 and g[(j+1)%len(g)] == 0):
            ng[j] = int(codigo[5])
        elif (g[j-1] == 0 and g[j] == 1 and g[(j+1)%len(g)] == 1):
            ng[j] = int(codigo[4])
        elif (g[j-1] == 1 and g[j] == 0 and g[(j+1)%len(g)] == 0):
            ng[j] = int(codigo[3])
        elif (g[j-1] == 1 and g[j] == 0 and g[(j+1)%len(g)] == 1):
            ng[j] = int(codigo[2])
        elif (g[j-1] == 1 and g[j] == 1 and g[(j+1)%len(g)] == 0):
            ng[j] = int(codigo[1])
        elif (g[j-1] == 1 and g[j] == 1 and g[(j+1)%len(g)] == 1):
            ng[j] = int(codigo[0])
    g = ng.copy() # se nao usar copy ambos vetores tornam-se o mesmo
    
plt.figure(1)
plt.axis('off')
plt.imshow(matriz_evolucao, cmap='gray')
plt.savefig('Automata.png', dpi=300)
plt.show()