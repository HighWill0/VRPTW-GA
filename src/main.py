import time
from utils.utils import *
from tqdm import tqdm

def algo_genetique_with_time_limit(data, population_size, distance_weight, waiting_time_weight, vehicules_number_weigth, crossover_rate, mutation_rate, elitism_rate, time_limit):
    print("Starting Genetic Algorithm with time limit...")
    print("Initializing population...")
    population = initialize_population(population_size, data)
    
    print("Population initialized.")
    print("Population list:", population)
    start_time = time.time()
    best_solutions = []
    iteration = 0

    with tqdm(total=time_limit, desc="Running Genetic Algorithm") as pbar:
        while True:
            iteration += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                break  # Exit the loop if time limit is reached
            
            pbar.update(elapsed_time - pbar.n)  # Update tqdm progress bar
            
            fitness_scores = [calculate_fitness(solution, data) for solution in population]
            
            parent1, parent1_number = roulette_selection(population, fitness_scores)
            parent2, parent2_number = roulette_selection(population, fitness_scores)
            
            if will_crossover(crossover_rate):
                ch1, ch2 = cx_partially_matched(parent1, parent2, data)
                
            else:
                ch1 = parent1
                ch2 = parent2

            mutated_solution = mutation_with_rate(list(ch1), data, mutation_rate)
            mutated_solution2 = mutation_with_rate(list(ch2), data, mutation_rate)
            nouvelle_population = replacement_with_elitism(population, [mutated_solution, mutated_solution2], elitism_rate, data, distance_weight, waiting_time_weight)
            population = nouvelle_population
            
            sol = display_result(population, vehicules_number_weigth)
            best_solutions.append(sol)
    
    best_solution(best_solutions)
    return population
    
def best_solution(solutions):


    nb_vehicule = solutions[0]['Number_of_Vehicles']
    min_total_distance = solutions[0]['Total_Distance']
    min_total_waiting_time = solutions[0]['Total_Waiting_Time']
    min_score = solutions[0]['Score']
    
    
    for i,solution in enumerate(solutions): 
        if((solution["Number_of_Vehicles"] < nb_vehicule) and (solution["Score"] < min_score) ):
            
            nb_vehicule = solution["Number_of_Vehicles"]
            min_score = solution["Score"]
            min_total_distance = solution["Total_Distance"]
            min_total_waiting_time = solution["Total_Waiting_Time"]
    
    print("Best Solution: ", solution['solution'] )        
    print(f"Solution: Score: {min_score}, Total Distance: {min_total_distance}, Total Waiting Time: {min_total_waiting_time}, Number of Vehicles: {nb_vehicule}")

def display_result(population, vehicules_number_weigth):
    nb_vehicule = 500
    min_total_distance=999999
    min_total_waiting_time = 999999
    min_score = 999999
    best_sol = []
    for i, solution in enumerate(population):  
        if not is_solution_valid(solution, data):
            continue
        score, total_distance, total_waiting_time, number_of_vehicles = objective(solution, data, vehicules_number_weigth)
        if(number_of_vehicles < nb_vehicule and score < min_score):
            nb_vehicule = number_of_vehicles
            min_score = score
            min_total_distance = total_distance
            min_total_waiting_time = total_waiting_time
            best_sol = solution
    bestSol= {
        'Score': min_score,
        'Total_Distance': min_total_distance,
        'Total_Waiting_Time': min_total_waiting_time,
        'Number_of_Vehicles': nb_vehicule,
        'solution': best_sol
        
    }
    return bestSol
         


data = start('C101')
population_size = 10
time_limit = 3600
nbr_iteration = 0

distance_weight = 0.65 # selection
waiting_time_weight = 0.65 # selection
vehicules_number_weight=0.43 # fonction objective
mutation_rate = 0.4 #0.05
crossover_rate= 0.5 #0.8
elitism_rate = 0.1


algo_genetique_with_time_limit(data, population_size, distance_weight, waiting_time_weight, vehicules_number_weight,
                               crossover_rate, mutation_rate, elitism_rate, time_limit)       
