import io
from json import dump, load
import math
from operator import attrgetter
import os
import random
#from util import make_directory_for_file, exist, load_instance, merge_rules

BASE_DIR = "/Users/hungle/VScode/Repo/VRPTW-GA/"## the path of the 'VRPTW' directory


def start(instance_name):
 
    json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    data = load_instance(json_file=json_file)

    if data is None:
        print("Data not found")
        return
    return data
    
def calculate_distance_data(customer1, customer2):
    aux = (customer1['coordinates']['x'] - customer2['coordinates']['x'])**2 + \
        (customer1['coordinates']['y'] - customer2['coordinates']['y'])**2
    return math.sqrt(aux)

def calculate_distance(customer1, customer2,data):
    aux = (data[customer1]['coordinates']['x'] - data[customer2]['coordinates']['x'])**2 + \
        (data[customer1]['coordinates']['y'] - data[customer2]['coordinates']['y'])**2
    return math.sqrt(aux)
    
            
def make_directory_for_file(path_name):
    try:
        os.makedirs(os.path.dirname(path_name))
    except OSError:
        pass

def load_instance(json_file):
    if exist(path=json_file):
        with io.open(json_file, 'rt', encoding='utf-8', newline='') as file_object:
            return load(file_object)
    return None

def exist(path):
    if os.path.exists(path):
        return True
    return False

def merge_rules(rules):
    is_fully_merged = True
    for round1 in rules:
        if round1[0] == round1[1]:
            rules.remove(round1)
            is_fully_merged = False
        else:
            for round2 in rules:
                if round2[0] == round1[1]:
                    print(f"merging {round1} and {round2}")
                    rules.append((round1[0], round2[1]))
                    rules.remove(round1)
                    rules.remove(round2)
                    is_fully_merged = False
    return rules, is_fully_merged

def getClientName(client_id):
    return f'customer_{client_id}'
def getClientId(client_name):
    return int(client_name.split('_')[1])


def select_random_client(clients_non_assigned):
    return random.choice(clients_non_assigned)

def getVehicleNumber(solution):
    return len(solution)

def calculate_total_distance(solution, data) -> list[int]:
    distance = 0
    for route in solution:
        # Insert depot at the start and end of the route
        route_with_depot = [0] + route + [0]
        for i in range(len(route_with_depot) - 1):
            distance += calculate_distance(getClientName(route_with_depot[i]), getClientName(route_with_depot[i + 1]), data)
    return distance


def calculate_waiting_time_per_route(route, data):
    total_waiting_time = 0
    current_time = 0
    if len(route) <= 1:
        current_client_id = route[0] if route else 0
        current_client = getClientName(current_client_id)

        arrival_time = data["distance_matrix"][0][current_client_id]  

        waiting_time = max(0, data[current_client]['ready_time'] - arrival_time)

        return waiting_time
    for client in route[1:]:
        current_client_id = client
        current_client = getClientName(current_client_id)
        
        #Calculate the arrival time at the current client. It is the sum of the current time and the time taken to travel from the previous client to the current one 
        arrival_time = current_time + data["distance_matrix"][current_client_id -1][current_client_id ]
        #waiting_time: Calculate the waiting time at the current client. It is the maximum of 0 and the difference between the start of the time window for the client (data["time_windows"][client][0]) and the calculated arrival time
        waiting_time = max(0, data[current_client]['ready_time'] - arrival_time) #exemple arrival fi 6 w houwa start=7 max bin 0 et 7-6=1 """" [0 ]start
        #If the vehicle arrives within the time window or after it starts, the waiting time is 0. Otherwise, it's the difference between the start of the time window and the arrival time.
        total_waiting_time += waiting_time
        current_time = max(arrival_time, data[current_client]['ready_time']) + data[current_client]['service_time']
        #Update the current time. It is the maximum of the arrival time and the start of the time window for the client, plus the service time required at the current client.
       
    
    return total_waiting_time


def calculate_waiting_time(solution, data) -> float:
    total_waiting_time = 0
    for route in solution:
        # Insert depot at the start and end of the route
        route_with_depot = [0] + route + [0]
        total_waiting_time += calculate_waiting_time_per_route(route_with_depot, data)
    return total_waiting_time


# Fonction pour calculer le score de fitness d'une solution (distance totale parcourue)
def calculate_fitness(solution, data, distance_weight=0.7, waiting_time_weight=0.3) -> float:
    total_distance = calculate_total_distance(solution, data)
    total_waiting_time = calculate_waiting_time(solution, data)

    scores = distance_weight * total_distance + waiting_time_weight * total_waiting_time 
    
    # Return the sum of scores for the entire solution
    return scores


def objective(solution, data,vehicules_number_weigth=0.7):
    total_distance=0
    total_waiting_time=0
    nombre_de_véhicules = len(solution)  # Nombre de routes dans la solution

    total_waiting_time += calculate_waiting_time(solution, data)
    total_distance += calculate_total_distance(solution, data)


    score=calculate_fitness(solution, data,distance_weight=0.3, waiting_time_weight=0.2)
    score=score+nombre_de_véhicules*vehicules_number_weigth

    return score,total_distance,total_waiting_time,nombre_de_véhicules



def initialize_population(population_size,data) -> list[list[list]]:
    population = []
    retry_count = 0
    max_tries = 5
    while len(population) < population_size:
        if retry_count >= max_tries:
            print("Max retries reached, stopping population generation.")
            break
        solution = generate_feasible_solution(data)
        if solution not in population:  # Ensure uniqueness
            population.append(solution)
        retry_count += 1
    return population

def generate_feasible_solution(data) -> tuple[list[list], int]:
    visited = [False] * len(data["distance_matrix"][0]) 
    visited[0] = True    
    total_cost = 0
    solution = []
    while len(solution) < len(data["distance_matrix"][0]) - 1: 
        
        route = []
        route.append(0)
        current_id = 0
        time = 0
        cost = 0
        demand = 0
        while True:
            next = -1
            best = 1e9
            indices = list(range(1, len(data["distance_matrix"][0])))
            random.shuffle(indices)  # Shuffle indices to start with a random order
            for i in indices:
                if not visited[i]:  
                    
                    client_current = getClientName(current_id)
                    client_candidat = getClientName(i)
                    arrival = time + data['distance_matrix'][current_id][i]
                    
                    if (
                        arrival <= data[client_candidat]['due_time']
                        and arrival >= data[client_candidat]['ready_time']
                        and data['distance_matrix'][current_id][i] < best
                        and demand + data[client_candidat]['demand'] <= data['vehicle_capacity']
                    ):
                        next = i
                        best = data['distance_matrix'][current_id][i]
                    
                   
            if next == -1:
                break
            visited[next] = True
            route.append(next)
            time += best + data[getClientName(next)]['service_time']
            cost += best
            demand += data[getClientName(next)]['demand']
            current_id = next
        cost += data['distance_matrix'][current_id][0]  # Return to depot
        total_cost += cost

        if len(route) == 1:  # If no clients were visited, break
            if visited.count(False) > 0:
                route = optimizePathForRedundentClients(visited, data)
            else:
                break
        route.append(0)  # Return to depot
        solution.append(route)
        
    
    return clean_route(solution)

def optimizePathForRedundentClients(visited, data):
    route = []
    route.append(0)  # Start from depot
    current_id = 0
    time = 0
    cost = 0
    demand = 0

    while True:
        unvisited_clients = [
            i for i in range(1, len(data["distance_matrix"][0])) if not visited[i]
        ]
        if not unvisited_clients:
            break

        # Find the nearest unvisited client
        nearest_client = min(
            unvisited_clients,
            key=lambda i: data["distance_matrix"][current_id][i]
        )
        client_candidat = getClientName(nearest_client)
        arrival = time + data["distance_matrix"][current_id][nearest_client]

        # If arrival is less than ready_time, wait until ready_time
        if arrival < data[client_candidat]["ready_time"]:
            arrival = data[client_candidat]["ready_time"]

        # Add the customer to the route
        route.append(nearest_client)
        visited[nearest_client] = True
        time = arrival + data[client_candidat]["service_time"]
        cost += data["distance_matrix"][current_id][nearest_client]
        demand += data[client_candidat]["demand"]
        current_id = nearest_client

    # Return to depot
    return route
    
def clean_route(solution):
    #remove depot (0) from the start and end of each route
    cleaned_solution = []
    for route in solution:
        if route[0] == 0 and route[-1] == 0:
            cleaned_route = route[1:-1]
        else:
            cleaned_route = route
        if cleaned_route:  # Ensure the route is not empty
            cleaned_solution.append(cleaned_route)
    
    return cleaned_solution

def generate_feasible_solution_nearest(data):
    solution = []
    vehicule = [0]
    capacity_current = 0
    time_current = 0

    clients_non_assigned = list(range(1, len(data["distance_matrix"][0]) ))  # 1 to 100

    while clients_non_assigned:
        client_current_id = vehicule[-1] if vehicule else 0  # Start from the depot for each vehicle
        selected_client_id = select_random_client(clients_non_assigned)
        selected_client = getClientName(selected_client_id)

        # Find the best position to insert the selected client in the route
        best_position = find_best_position(selected_client_id, vehicule, data)

        # Check if adding the client to the route violates constraints
        if (
            capacity_current + data[selected_client]["demand"] <= data["vehicle_capacity"]
            and time_current
            + data["distance_matrix"][client_current_id][selected_client_id]
            <= data[selected_client]["due_time"]
        ):

            # Insert the client into the route at the best position
            vehicule.insert(best_position, selected_client_id)
            capacity_current += data[selected_client]["demand"]
            time_current += (
                data["distance_matrix"][client_current_id][selected_client_id]
                + data[selected_client]["service_time"]
            )

            # Remove the client from the list of non-assigned clients
            clients_non_assigned.remove(selected_client_id)

        else:
            if vehicule[-1] != 0:
                vehicule.append(0)
                solution.append(vehicule[:])
            vehicule = [0]
            capacity_current = 0
            time_current = 0

    return solution
def find_best_position(selected_client_id, route, data):
    best_position = 1  # Start from the first position

    # Initialize variables for the best insertion cost and the current route length
    best_insertion_cost = float('inf')
    current_route_length = calculate_total_distance([route], data)

    # Iterate over possible positions in the route
    for position in range(1, len(route)):
        # Calculate the cost of inserting the client at the current position
        insertion_cost = calculate_insertion_cost(selected_client_id, position, route, data)

        # Check if the insertion is feasible and improves the route length
        if insertion_cost < best_insertion_cost:
            best_insertion_cost = insertion_cost
            best_position = position

    return best_position


def calculate_insertion_cost(selected_client_id, position, route, data):
    # Calculate the change in route length by inserting the selected client at the specified position
    prev_client_id = route[position - 1]
    next_client_id = route[position]
    selected_client = getClientName(selected_client_id)

    insertion_cost = (
        data["distance_matrix"][prev_client_id][selected_client_id]
        + data["distance_matrix"][selected_client_id][next_client_id]
        - data["distance_matrix"][prev_client_id][next_client_id]
    )

    # Check if the insertion satisfies time window constraints
    arrival_time = calculate_arrival_time(selected_client_id, prev_client_id, route, data)
    waiting_time = max(0, data[selected_client]["ready_time"] - arrival_time)
    time_penalty = waiting_time + data[selected_client]["service_time"]

    # Include the time penalty in the insertion cost
    insertion_cost += time_penalty

    return insertion_cost


def calculate_arrival_time(selected_client_id, prev_client_id, route, data):
    # Calculate the arrival time at the selected client in the route
    prev_client_index = route.index(prev_client_id)
    travel_time = data["distance_matrix"][route[prev_client_index]][selected_client_id]
    arrival_time = route_time_at_index(prev_client_index, route, data) + travel_time
    return arrival_time


def route_time_at_index(index, route, data):
    # Calculate the total time spent on the route up to the specified index
    time = 0
    for i in range(index):
        current_client_id = route[i]
        next_client_id = route[i + 1]
        time += data["distance_matrix"][current_client_id][next_client_id] + data[getClientName(current_client_id)]["service_time"]
    return time

def initialize_population_insertion(population_size, data):
    population = []

    while len(population) < population_size:
        solution = generate_feasible_solution_nearest(data)

        # Check if the solution is unique in the population
        if solution not in population:
            population.append(solution)

    return population


def roulette_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    # print(f"finess score: {fitness_scores}")
    selected_index = None

    random_value = random.uniform(0, total_fitness)
    
    cumulative_fitness = 0
    for i, score in enumerate(fitness_scores):
        cumulative_fitness += score
        if cumulative_fitness >= random_value:
            selected_index = i
            break

    # Return the selected individual from the population
    selected_individual = population[selected_index]
    return selected_individual ,selected_index


# mutation 
def mutation_inversion(solution, data):
    
    mutated_solution = solution.copy()  
    
    
    random_route_index = random.randint(0, len(mutated_solution) - 1)
    random_route = mutated_solution[random_route_index]
    

    if len(random_route) > 2:
        idx1, idx2 = random.sample(range(1, len(random_route)), 2)
        idx1, idx2 = sorted([idx1, idx2])
    else:
        return solution.copy()  # Return original solution if mutation is not possible
    
    random_route[idx1:idx2+1] = reversed(random_route[idx1:idx2+1])
    
    if not is_solution_valid(list(mutated_solution), data):
        mutated_solution = solution.copy()
    
    return mutated_solution



def is_solution_valid(solution, data):
    solutions = list(solution)
    for route in solutions:
        capacity = 0
        time_current = 0
        
        if route[0] != 0 or route[-1] != 0:
            route.insert(0, 0)  # Ensure depot at the start
            route.append(0)  # Ensure depot at the end
        if 0 in route[1:len(route) - 1]:
            return False 

        for i in range(len(route) - 1):
            client_current_id = route[i]
            client_candidat_id = route[i + 1]

            client_current = getClientName(client_current_id)
            client_candidat = getClientName(client_candidat_id)

            demand = data[client_candidat]['demand']
            service_time = data[client_candidat]['service_time']
            
            ready_time = data[client_candidat]['ready_time']
            due_time = data[client_candidat]['due_time']
            
            # Check if adding the client to the route violates constraints
            if (capacity + demand <= data['vehicle_capacity'] and
                    time_current + data['distance_matrix'][client_current_id][client_candidat_id] <= due_time):

                # Update the capacity and time for the current vehicle
                capacity += demand
                time_current += data['distance_matrix'][client_current_id][client_candidat_id] + service_time

            else:

                    return False

    return True

def mutation_with_rate(solution, data, mutation_rate):
    mutated_solution = mutation_with_rate1(solution, data, mutation_rate)
    if not is_solution_valid(mutated_solution, data):
        return solution
    else:
        return mutated_solution
   
    
    
    return mutated_solution
def mutation_with_rate1(solution, data, mutation_rate):
    mutated_solution = solution.copy()
    
    # Verification of the mutation rate
    if random.random() < mutation_rate:
        # Apply the inversion mutation with the mutation_inversion function
        mutated_solution = mutation_inversion(mutated_solution, data)
    
    return mutated_solution
    

def cx_partially_matched(ind1, ind2, data):
    
    child1, child2 = cx_partially_matched1(ind1, ind2)
    
    if not is_solution_valid(child1, data):
        child1 = ind1
    if not is_solution_valid(child2, data) :
        child2= ind2
   
    return child1, child2
    
  
def cx_partially_matched1(ind1, ind2):
    """
    PMX crossover at the route level (list of list[int]).
    If a route has only 1 customer, merge it with the next route (or previous if at the end).
    """
    def merge_single_route(routes, idx):
        # Merge route at idx (len==1) with next or previous route
        if len(routes) <= 1:
            return routes
        if idx < len(routes) - 1:
            # Merge with next
            merged = routes[idx] + routes[idx + 1]
            return routes[:idx] + [merged] + routes[idx + 2:]
        else:
            # Merge with previous
            merged = routes[idx - 1] + routes[idx]
            return routes[:idx - 1] + [merged]
    
    # Copy to avoid modifying originals
    routes1 = [list(r) for r in ind1]
    routes2 = [list(r) for r in ind2]

    # Merge single-customer routes in both parents
    for routes in (routes1, routes2):
        i = 0
        while i < len(routes):
            if len(routes[i]) == 1:
                routes[:] = merge_single_route(routes, i)
                # After merge, don't increment i (list shrinks)
            else:
                i += 1


    min_len = min(len(routes1), len(routes2))
    if min_len < 2:
        return routes1, routes2 

    cxpoint1, cxpoint2 = sorted(random.sample(range(min_len), 2))


    part1 = routes2[cxpoint1:cxpoint2+1]
    part2 = routes1[cxpoint1:cxpoint2+1]

    child1 = routes1[:cxpoint1] + part1 + routes1[cxpoint2+1:]
    child2 = routes2[:cxpoint1] + part2 + routes2[cxpoint2+1:]

    return child1, child2

def replacement_with_elitism(population, enfants, elitism_rate, data, distance_weight=0.7, waiting_time_weight=0.3):
    # Population size
    population_size = len(population)
    
    # Number of elites to keep
    nombre_elites = round(elitism_rate * population_size)
    
    # Internal fitness function using calculate_fitness with the given parameters
    def fitness_function(solution):
        return calculate_fitness(solution, data, distance_weight, waiting_time_weight)
    
    # Calculation of fitness scores for the current population and the children
    population_fitness = [fitness_function(solution) for solution in population]
    enfants_fitness = [fitness_function(enfant) for enfant in enfants]
    
    # Combination of individuals from the current population and children
    individus_combines = population + enfants
    individus_fitness_combines = population_fitness + enfants_fitness
    
    # Sorting of combined individuals based on their fitness score
    individus_tries = [individu for _, individu in sorted(zip(individus_fitness_combines, individus_combines), reverse=True)]
    
    # Selection of the elites
    elites = individus_tries[:nombre_elites]
    
    # Training the new population by selecting the best individuals (excluding the elites)
    nouvelle_population = individus_tries[nombre_elites:]
    
    # Addition of the elites from the initial population to the new population
    for elite_solution in population[:nombre_elites]:
        nouvelle_population.append(elite_solution)
    
    return nouvelle_population




def order_crossover(ind1, ind2, data):
    child1, child2 = order_crossover1(ind1, ind2)
    while not (is_solution_valid(child1, data) ) and (not is_solution_valid(child2,data) ) :
         child1, child2 = order_crossover1(ind1, ind2)
    return child1, child2

def order_crossover1(parent1, parent2):
    # Choice of two crossover points
    crossover_points = sorted(random.sample(range(min(len(parent1), len(parent2))), 2))

    # Copy of the subsequences between the crossover points
    subsequence_parent1 = parent1[crossover_points[0]:crossover_points[1] + 1]
    subsequence_parent2 = parent2[crossover_points[0]:crossover_points[1] + 1]

    # Initialization of children with lists containing markers
    child1 = [None] * len(parent1)
    child2 = [None] * len(parent2)

    # Copy of subsequences in the children
    child1[crossover_points[0]:crossover_points[1] + 1] = subsequence_parent1
    child2[crossover_points[0]:crossover_points[1] + 1] = subsequence_parent2

    # Filling in the gaps in children
    fill_gaps(child1, parent2, crossover_points)
    fill_gaps(child2, parent1, crossover_points)

    return list(child1), list(child2)



def fill_gaps(child, parent, crossover_points):
    # Filling the gaps with the parent elements not included in the subsequence
    position = crossover_points[1] + 1

    for element in parent:
        if element not in child:
            while position >= len(child):  # Ensure position is within bounds
                position -= len(child)
            child[position] = element
            position += 1


# A higher value increases the likelihood of crossover, while a lower value decreases it. 
def will_crossover(crossover_rate=0.8):
    return random.random() <crossover_rate