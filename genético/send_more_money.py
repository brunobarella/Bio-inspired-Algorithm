import random
import matplotlib.pyplot as plt
import numpy as np
import collections

# Definição do problema
variables = ["s", "e", "n", "d", "m", "o", "r", "y"]
problem = ["send", "more", "money"]
letters = [letter for word in problem for letter in word]

# Hiperparâmetros do algoritmo genético
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.05

# Função para gerar um indivíduo aleatório
def generate_individual():
    return random.sample(list(range(10)), len(variables))

# Função para avaliar a aptidão de um indivíduo
def fitness(individual):
    mapping = dict(zip(variables, individual))
    values = [int("".join([str(mapping[letter]) for letter in word])) for word in problem]
    return abs(values[0] + values[1] - values[2])

def roulette_selection(population, fitness_scores):
    # cria uma lista com as probabilidades de seleção de cada indivíduo
    selection_probabilities = [score/sum(fitness_scores) for score in fitness_scores]
    # seleciona dois pais aleatoriamente com base nas probabilidades
    parent = random.choices(population, weights=selection_probabilities)[0]
    return parent


# Função para realizar o cruzamento de dois pais
# veficar a repetição
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        index = random.randint(0, len(parent1))
        child1 = parent1[:index] + parent2[index:]
        child2 = parent2[:index] + parent1[index:]
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2


def cyclic_crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        # obter o comprimento das palavras e garantir que são iguais
        n = len(parent1)
        assert n == len(parent2)

        # escolher aleatoriamente um ponto inicial
        start = random.randrange(n)
        
        lack_p1 = [i for i in range(10) if i not in parent1]
        lack_p2 = [i for i in range(10) if i not in parent2]
        
        parent1 =parent1+lack_p1
        parent2 =parent2+lack_p2

        i = start    
        value_i_p1 = parent1[i]
        parent1[i] = parent2[i]
        parent2[i] = value_i_p1
        
        iteration = 0
        while len([item for item, count in collections.Counter(parent1).items() if count > 1])>0:#start!=i or iteration==0:
            last_i = i 
            i = [idx for idx, value in enumerate(parent1) if value == parent1[i]]
            i = [x for x in i if x!=last_i][0]
            value_i_p1 = parent1[i]
            parent1[i] = parent2[i]
            parent2[i] = value_i_p1
            iteration += 1

        if len(parent1)!=n:
            for i in range(len(parent1)-n):
                parent1.pop(9-i)
                parent2.pop(9-i)
    
    return parent1, parent2


# Função para mutar um indivíduo
def mutate(individual):
    if random.random() < MUTATION_RATE:
        index1, index2 = random.sample(range(len(individual)), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

def ordered_reinsertion(population, fitnesses):#, parents, children
    # ordene a população atual em ordem decrescente de fitness
    sorted_population = [p for _, p in sorted(zip(fitnesses, population), reverse=False)]
    # # selecione os pais da população anterior que geraram os filhos
    # selected_parents = sorted_population[:len(children)]
    # # adicione os filhos à população atual
    # new_population = sorted_population + children
    # # ordene a nova população em ordem decrescente de fitness
    # sorted_new_population = [p for _, p in sorted(zip([fitness(p) for p in new_population], new_population), reverse=True)]
    # # remova os indivíduos menos aptos da nova população
    # final_population = sorted_new_population[:len(population)-len(selected_parents)]
    # # adicione os pais selecionados à população final
    # final_population += selected_parents
    
    final_population = sorted_population[:int(POPULATION_SIZE*0.4)]
    return final_population

# Função para executar o algoritmo genético
def genetic_algorithm():
    # Gerar população inicial
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # fitnesses = [fitness(individual) for individual in population]
    # list(filter(lambda score: score ==1, [row[5] for row in population]))
    # list(filter(lambda score: score <1000, fitnesses))
    generations = 0

    pais = []
    filhos = []
    while generations < MAX_GENERATIONS:
        # Avaliar a aptidão de cada indivíduo
        fitnesses = [fitness(individual) for individual in population]

        # Verificar se a solução foi encontrada
        best_fitness = min(fitnesses)
        if best_fitness == 0:
            best_individual = population[fitnesses.index(best_fitness)]
            mapping = dict(zip(variables, best_individual))
            solution = f"{mapping['s']}{mapping['e']}{mapping['n']}{mapping['d']} + " \
                       f"{mapping['m']}{mapping['o']}{mapping['r']}{mapping['e']} = " \
                       f"{mapping['m']}{mapping['o']}{mapping['n']}{mapping['e']}{mapping['y']} "\
                       f" fitness = {best_fitness}"
            return solution, pais, filhos

        population2 = []
                
        for _ in range(int(POPULATION_SIZE*0.6)):
        
            # Selecionar dois pais usando a roleta
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
            
            pais.append(fitness(parent1))
            pais.append(fitness(parent2))

            # Cruzar os pais para gerar dois filhos
            child1, child2 = cyclic_crossover(parent1, parent2)

            # Mutar os filhos
            child1 = mutate(child1)
            child2 = mutate(child2)

            # Adicionar os filhos à nova população
            if child1 not in population2:
                population2.append(child1)
                filhos.append(fitness(child1))
            if child2 not in population2:
                population2.append(child2)
                filhos.append(fitness(child2))
            
        # Reinserção Ordenada: população total é ordenada e selecionam-se os Tp melhores
        population_2 = ordered_reinsertion(population, fitnesses)
            
        
        population = population2+population_2
        
        print(f"Geração: {generations}")
        # Se o número máximo de gerações for alcançado, retornar a melhor solução encontrada
        fitnesses = [fitness(individual) for individual in population]
        best_fitness = min(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        mapping = dict(zip(variables, best_individual))
        solution = f"{mapping['s']}{mapping['e']}{mapping['n']}{mapping['d']} + " \
                f"{mapping['m']}{mapping['o']}{mapping['r']}{mapping['e']} = " \
                f"{mapping['m']}{mapping['o']}{mapping['n']}{mapping['e']}{mapping['y']} "\
                f" fitness = {best_fitness}"
        print(solution)
        generations += 1
        
    # Se o número máximo de gerações for alcançado, retornar a melhor solução encontrada
    fitnesses = [fitness(individual) for individual in population]
    best_fitness = min(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    mapping = dict(zip(variables, best_individual))
    solution = f"{mapping['s']}{mapping['e']}{mapping['n']}{mapping['d']} + " \
               f"{mapping['m']}{mapping['o']}{mapping['r']}{mapping['e']} = " \
               f"{mapping['m']}{mapping['o']}{mapping['n']}{mapping['e']}{mapping['y']} "\
               f" fitness = {best_fitness}"

    return solution, pais, filhos

if __name__ == '__main__':
    solution, pais, filhos = genetic_algorithm()
    print(solution)

    plt.plot(pais)
    plt.show()
    plt.plot(filhos)
    plt.show()