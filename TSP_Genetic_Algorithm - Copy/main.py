import pandas as pd  # Import the pandas library to work with data tables easily
import numpy as np   # Import the numpy library for fast mathematical operations
import math          # Import the math library for mathematical functions like square root
import random        # Import the random library to generate random numbers for the algorithm
import matplotlib.pyplot as plt # Import the pyplot module from matplotlib to draw graphs
import os            # Import the os library to interact with the operating system (e.g., check files)
import shutil        # Import the shutil library to manage files and directories (like deleting folders)

# ==========================================
# PART 1: Parser and Distance Calculation
# ==========================================

def parse_tsp_file(filename):
    """
    Reads the .tsp file and extracts city coordinates.
    """
    cities = [] # Initialize an empty list to store the city data extracted from the file
    
    # Open the specified file in read mode ('r')
    with open(filename, 'r') as file:
        lines = file.readlines() # Read all lines from the file and store them in a list

    start_reading = False # Create a flag to track when the coordinate section starts
    
    # Loop through each line in the file content
    for line in lines:
        # Check if the current line marks the start of the coordinate section
        if "NODE_COORD_SECTION" in line:
            start_reading = True # Set the flag to True to start processing data
            continue # Skip the rest of this loop iteration and go to the next line
        
        # Check if the current line marks the end of the file
        if "EOF" in line:
            break # Exit the loop as we have finished reading the data
            
        # If the flag is True (we are in the data section) and the line is not empty
        if start_reading and line.strip():
            parts = line.strip().split() # Remove whitespace and split the line into parts by spaces
            
            # Check if the line has at least 3 parts (ID, X, Y) to be valid
            if len(parts) >= 3:
                city_id = int(parts[0])   # Convert the first part to an integer (City ID)
                x = float(parts[1])       # Convert the second part to a float (X coordinate)
                y = float(parts[2])       # Convert the third part to a float (Y coordinate)
                cities.append([city_id, x, y]) # Add the city data as a list to the 'cities' list

    # Create a pandas DataFrame from the cities list with column names 'city_id', 'x', and 'y'
    df = pd.DataFrame(cities, columns=['city_id', 'x', 'y'])
    
    # Extract the 'x' and 'y' columns as a NumPy array for faster calculation
    coords = df[['x', 'y']].values  
    
    # Convert the 'city_id' column to a standard Python list
    city_ids = df['city_id'].tolist()
    
    # Create a dictionary to map each City ID to its zero-based index in the array
    id_to_index = {city_id: idx for idx, city_id in enumerate(city_ids)}
    
    # Return the coordinates, the list of city IDs, and the ID mapping
    return coords, city_ids, id_to_index

def calculate_distance(city1, city2):
    """
    Calculates Euclidean distance between two cities.
    Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
    """
    # Calculate the squared difference in X coordinates
    # Add the squared difference in Y coordinates
    # Take the square root of the sum to get the distance
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# ==========================================
# PART 2: Solution Representation and Greedy
# ==========================================

def create_random_solution(city_ids):
    """
    Creates a random valid tour (solution).
    """
    solution = city_ids.copy() # Create a copy of the city IDs list to avoid modifying the original
    random.shuffle(solution)   # Randomly shuffle the order of cities in the list
    return solution            # Return the randomized list as a solution

def calculate_fitness(solution, cities_df):
    """
    Calculates the total distance of a tour.
    In this problem, 'Fitness' = 'Total Distance'.
    LOWER distance is BETTER.
    """
    # Unpack the cities_df tuple into coordinates, (unused city_ids), and the index map
    coords, _, id_to_index = cities_df
    
    total_distance = 0       # Initialize the total distance counter to zero
    n = len(solution)        # Get the number of cities in the solution
    
    # Loop through every city in the solution path
    for i in range(n):
        # Get the array index of the current city using the map
        idx1 = id_to_index[solution[i]]
        
        # Get the array index of the next city
        # The modulo operator (%) ensures the last city connects back to the first one
        idx2 = id_to_index[solution[(i + 1) % n]]
        
        # Retrieve the coordinates for the current city
        city1 = coords[idx1]
        # Retrieve the coordinates for the next city
        city2 = coords[idx2]
        
        # Calculate the distance between them and add it to the total
        total_distance += calculate_distance(city1, city2)
        
    return total_distance # Return the final total distance

def greedy_algorithm(cities_df, start_city):
    """
    Greedy Approach (Nearest Neighbor).
    Always chooses the closest unvisited city.
    Used to create good initial solutions.
    """
    # Unpack the data structures
    coords, city_ids, id_to_index = cities_df
    
    # Create a set of unvisited cities for fast lookup and removal
    unvisited = set(city_ids)
    
    current_city = start_city      # Set the starting city
    unvisited.remove(current_city) # Remove the start city from the unvisited set
    solution = [current_city]      # Initialize the solution list with the start city
    
    # Loop as long as there are still unvisited cities
    while unvisited:
        # Get the index of the current city
        idx_current = id_to_index[current_city]
        # Get the coordinates of the current city
        current_coords = coords[idx_current]
        
        closest_city = None             # Variable to store the closest city found so far
        min_distance = float('inf')     # Set initial minimum distance to infinity
        
        # Iterate through all remaining unvisited cities
        for city in unvisited:
            idx_city = id_to_index[city]       # Get index of the candidate city
            city_coords = coords[idx_city]     # Get coordinates of the candidate city
            
            # Calculate distance from current city to candidate city
            dist = calculate_distance(current_coords, city_coords)
            
            # If this distance is smaller than the current minimum
            if dist < min_distance:
                min_distance = dist    # Update the minimum distance
                closest_city = city    # Update the closest city
        
        # Add the found closest city to the solution path
        solution.append(closest_city)
        # Remove the found city from the unvisited set
        unvisited.remove(closest_city)
        # Update the current city to be the one we just moved to
        current_city = closest_city
        
    return solution # Return the completed greedy path

def print_solution_info(solution, cities_df, label=""):
    """
    Helper function to print solution details to console.
    """
    # Calculate the fitness (distance) of the given solution
    fitness = calculate_fitness(solution, cities_df)
    # Create a string representation of the solution list
    solution_str = " ".join(str(city) for city in solution)
    # Print the solution path prefixed with a label
    print(f"{label}Solution: {solution_str}")
    # Print the fitness score prefixed with a label
    print(f"{label}Fitness (total distance): {fitness:.2f}")
    return fitness # Return the fitness value

# ==========================================
# PART 3: Population and Selection/Crossover
# ==========================================

def create_initial_population(cities_df, population_size, include_greedy=False):
    """
    Creates the first generation of solutions.
    Can mix random solutions with greedy solutions for better start.
    """
    # Unpack the data structures
    coords, city_ids, id_to_index = cities_df
    population = [] # Initialize an empty list for the population
    
    # Check if we should include greedy solutions in the initial population
    if include_greedy:
        # Loop through the first 5 cities to use them as starting points
        for start_city in city_ids[:5]:
            # Generate a greedy solution starting from this city
            greedy_solution = greedy_algorithm(cities_df, start_city)
            # Add the greedy solution to the population
            population.append(greedy_solution)
            
    # Continue adding solutions until the population reaches the desired size
    while len(population) < population_size:
        # Generate a purely random solution
        random_solution = create_random_solution(city_ids)
        # Add the random solution to the population
        population.append(random_solution)
        
    return population[:population_size] # Return the population (trimmed to exact size if needed)

def tournament_selection(population, cities_df, tournament_size=3):
    """
    Selection Method: Tournament.
    Picks a few individuals randomly and selects the best one.
    Why? It preserves diversity better than just picking the absolute best.
    """
    # Randomly select 'tournament_size' individuals from the population
    tournament = random.sample(population, tournament_size)
    
    tournament_fitness = [] # List to store (fitness, individual) pairs
    
    # Loop through each individual in the tournament
    for individual in tournament:
        # Calculate the fitness of the individual
        fitness = calculate_fitness(individual, cities_df)
        # Add the pair to the list
        tournament_fitness.append((fitness, individual))
    
    # Sort the list by fitness (ascending order, so lowest distance comes first)
    tournament_fitness.sort() 
    
    # Return the individual with the best (lowest) fitness score
    return tournament_fitness[0][1]

def ordered_crossover(parent1, parent2):
    """
    Crossover Method: Ordered Crossover (OX1).
    Why? Standard crossover creates duplicates. This method ensures
    every city appears exactly once (valid permutation).
    """
    size = len(parent1) # Get the size of the chromosome (number of cities)
    
    # Pick two random cut points and sort them so point1 < point2
    point1, point2 = sorted(random.sample(range(size), 2))
    
    # Initialize the child with -1 (placeholder for empty spots)
    child = [-1] * size
    
    # Copy the segment between the cut points from Parent 1 to the Child
    child[point1:point2] = parent1[point1:point2]
    
    # Initialize an index to track position in Parent 2
    parent2_index = 0
    
    # Loop through the child to fill the remaining empty spots
    for i in range(size):
        # Check if the current spot is empty
        if child[i] == -1: 
            # Find the next city in Parent 2 that is NOT already in the child
            while parent2[parent2_index] in child:
                parent2_index += 1 # Move to the next city in Parent 2
            
            # Place the found city into the child
            child[i] = parent2[parent2_index]
    
    return child # Return the new child solution

# ==========================================
# PART 4: Mutation and Main GA Loop
# ==========================================

def swap_mutation(individual, mutation_prob):
    """
    Mutation Method 1: Swap.
    Randomly swaps two cities.
    Why? Adds small random changes to explore new solutions.
    """
    mutated = individual.copy() # Create a copy to avoid modifying the original individual
    
    # Loop through each position in the tour
    for i in range(len(mutated)):
        # Generate a random number and check if it's less than mutation probability
        if random.random() < mutation_prob:
            # Pick a random index j to swap with
            j = random.randint(0, len(mutated) - 1)
            # Swap the city at index i with the city at index j
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
    return mutated # Return the mutated individual

def inversion_mutation(individual, mutation_prob):
    """
    Mutation Method 2: Inversion.
    Reverses a section of the tour.
    Why? Better for TSP because it keeps edges connected (untangles knots).
    """
    mutated = individual.copy() # Create a copy of the individual
    
    # Check if mutation should happen based on random probability
    if random.random() < mutation_prob:
        n = len(mutated) # Get the length of the tour
        # Select two random indices for the start and end of the segment
        i, j = sorted(random.sample(range(n), 2))
        
        # Reverse the segment of the tour between index i and j
        # Python slicing [i:j] gets the segment, reversed() flips it
        mutated[i:j] = reversed(mutated[i:j])
        
    return mutated # Return the mutated individual

def create_new_generation(old_population, cities_df, crossover_prob, mutation_prob, mutation_type='swap'):
    """
    Creates the next generation (Epoch).
    Process: Selection -> Crossover -> Mutation.
    """
    new_population = [] # Initialize list for the new generation
    population_size = len(old_population) # Get the target size
    
    best_solution = None            # Variable to store the best solution found so far
    best_fitness = float('inf')     # Set initial best fitness to infinity
    
    # Elitism: Loop through old population to find the single best individual
    for individual in old_population:
        fitness = calculate_fitness(individual, cities_df) # Calculate fitness
        if fitness < best_fitness: # If this individual is better than the best found
            best_fitness = fitness # Update best fitness
            best_solution = individual # Update best solution
            
    # Add the best solution to the new population directly (Elitism)
    new_population.append(best_solution)
    
    # Continue creating new individuals until the population is full
    while len(new_population) < population_size:
        # 1. Selection: Select two parents using tournament selection
        parent1 = tournament_selection(old_population, cities_df)
        parent2 = tournament_selection(old_population, cities_df)
        
        # 2. Crossover: Decide whether to perform crossover based on probability
        if random.random() < crossover_prob:
            # Create a child by combining parents
            child = ordered_crossover(parent1, parent2)
        else:
            # If no crossover, the child is a copy of parent 1
            child = parent1.copy() 
            
        # 3. Mutation: Decide which mutation type to apply
        if mutation_type == 'inversion':
            # Apply inversion mutation
            child = inversion_mutation(child, mutation_prob)
        else:
            # Apply swap mutation
            child = swap_mutation(child, mutation_prob)
            
        # Add the created child to the new population
        new_population.append(child)
        
    # Return the new population, along with the best solution and its fitness
    return new_population, best_solution, best_fitness

def print_population_info(population, cities_df, generation):
    """
    Prints stats about the current generation.
    Useful to see if algorithm is improving.
    """
    # Create a list of fitness values for all individuals in the population
    fitness_values = [calculate_fitness(ind, cities_df) for ind in population]
    
    # Print the header for the current generation
    print(f"\nGeneration {generation} Statistics:")
    # Print the size of the population
    print(f"  Population size: {len(population)}")
    # Print the best (minimum) score found in this generation
    print(f"  Best score: {min(fitness_values):.2f}")     
    # Print the median score
    print(f"  Median score: {np.median(fitness_values):.2f}") 
    # Print the worst (maximum) score found
    print(f"  Worst score: {max(fitness_values):.2f}")    

def run_genetic_algorithm(cities_df, epochs=100, pop_size=50, 
                         crossover_prob=0.8, mutation_prob=0.1, mutation_type='swap'):
    """
    The Main Loop of Genetic Algorithm.
    Runs for a specific number of 'epochs' (generations).
    """
    # Print a message indicating start and mutation type
    print(f"Creating initial population... (Mutation: {mutation_type})")
    
    # Step 1: Initialize the population (optionally including greedy solutions)
    population = create_initial_population(cities_df, pop_size, include_greedy=True)
    
    best_fitness_history = []       # List to track best fitness per epoch
    best_solution_global = None     # Variable to store the best solution ever found
    best_fitness_global = float('inf') # Set global best fitness to infinity
    
    # Step 2: Evolution Loop - iterate for the given number of epochs
    for epoch in range(epochs):
        # Create the next generation using the previous one
        population, best_solution, best_fitness = create_new_generation(
            population, cities_df, crossover_prob, mutation_prob, mutation_type
        )
        
        # Append the best fitness of this epoch to history
        best_fitness_history.append(best_fitness)
        
        # Check if the best fitness of this epoch is better than the global best
        if best_fitness < best_fitness_global:
            best_fitness_global = best_fitness    # Update global best fitness
            best_solution_global = best_solution  # Update global best solution

        # Every 10 epochs, print population statistics
        if epoch % 10 == 0:
            print_population_info(population, cities_df, epoch)
            
    # Return the best global solution, its fitness, and the history of progress
    return best_solution_global, best_fitness_global, best_fitness_history

# ==========================================
# PART 5: Visualization and Comparisons
# ==========================================

def plot_results(fitness_history, instance_name):
    """
    Plots the progress of the Genetic Algorithm.
    X-axis: Epochs, Y-axis: Fitness (Distance).
    """
    plt.figure(figsize=(12, 7)) # Create a new figure with specified size
    # Plot the fitness history as a green line
    plt.plot(fitness_history, color='green', linewidth=2, label='Best ever')
    plt.xlabel('Epoch', fontsize=14)       # Set X-axis label
    plt.ylabel('Best Fitness', fontsize=14) # Set Y-axis label
    plt.title(f'{instance_name} - GA Progress', fontsize=16) # Set title
    plt.grid(True, alpha=0.3) # Enable grid lines with slight transparency
    
    # Define a dictionary of optimal scores for known instances
    optimal_dict = {'Berlin11': 4038, 'Berlin52': 7542, 'kroA100': 21282, 'kroA150': 26524}
    
    # If the current instance has a known optimal score
    if instance_name in optimal_dict:
        # Draw a red dashed horizontal line at the optimal score
        plt.axhline(optimal_dict[instance_name], color='red', linestyle='--', label=f'Optimal ({optimal_dict[instance_name]})')
        
    plt.legend() # Show the legend
    plt.tight_layout() # Adjust layout to prevent clipping
    # Save the figure to a file
    plt.savefig(f'results/{instance_name}_GA_progress.png')
    plt.close() # Close the figure to free up memory

def plot_comparison(optimal, ga, greedy, random_best, random_avg, instance_name):
    """
    Bar chart comparing different algorithms (GA vs Greedy vs Random).
    """
    # Define labels for the methods
    methods = ['Optimal', 'GA', 'Greedy', 'Random (best)', 'Random (avg)']
    # Define the scores corresponding to each method
    scores = [optimal, ga, greedy, random_best, random_avg]
    # Define colors for the bars
    colors = ['gold', 'green', 'blue', 'orange', 'red']
    
    # Create a subplot for the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create the bars
    bars = ax.bar(methods, scores, color=colors, edgecolor='black')
    
    # Loop through each bar to add text labels
    for bar in bars:
        yval = bar.get_height() # Get the height of the bar (the score)
        # Place text showing the score above the bar
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(scores)*0.02), f'{int(yval)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax.set_title(f'{instance_name} - Method Comparison', fontsize=16) # Set chart title
    ax.set_xlabel('Method', fontsize=14) # Set X-axis label
    ax.set_ylabel('Fitness (Distance)', fontsize=14) # Set Y-axis label
    plt.tight_layout() # Adjust layout
    plt.savefig(f'results/{instance_name}_comparison.png') # Save the chart
    plt.close() # Close figure

def plot_tour(solution, cities_df, fitness, instance_name):
    """
    Plots the map of cities and the path connecting them.
    """
    # Unpack city data
    coords, city_ids, id_to_index = cities_df
    
    # Create lists of X and Y coordinates for the tour
    # Add the first city to the end of the list to close the loop
    x = [coords[id_to_index[city]][0] for city in solution] + [coords[id_to_index[solution[0]]][0]]
    y = [coords[id_to_index[city]][1] for city in solution] + [coords[id_to_index[solution[0]]][1]]
    
    plt.figure(figsize=(10, 8)) # Create figure
    # Plot the tour path with lines and dots
    plt.plot(x, y, 'o-', color='blue', label='Tour path') 
    # Plot the cities as red dots
    plt.scatter(x[:-1], y[:-1], color='red', s=100) 
    
    # Loop through each city to add a label
    for i, city in enumerate(solution):
        # Place the city ID text near its coordinate
        plt.text(coords[id_to_index[city]][0], coords[id_to_index[city]][1], str(city), fontsize=10, fontweight='bold')
        
    # Set title including the fitness score
    plt.title(f'{instance_name} - Best Tour (Fitness: {fitness:.2f})', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=14) # Set X label
    plt.ylabel('Y Coordinate', fontsize=14) # Set Y label
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'results/{instance_name}_best_tour.png') # Save image
    plt.close() # Close figure

def plot_param_comparison(results_dict, instance_name):
    """
    Advanced plotting: Creates a 2x2 grid of charts showing how
    different parameters (population size, mutation rate etc.) affect results.
    """
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    # Set the main title for the entire figure
    fig.suptitle(f'{instance_name} - Parameter Comparison', fontsize=18)
    
    # 1. Population Size Chart (Top-Left)
    if 'pop_size' in results_dict:
        vals = results_dict['pop_size'] # Get data
        # Plot size vs score
        axs[0,0].plot([v[0] for v in vals], [v[1] for v in vals], marker='o', color='blue')
        # Add text labels for values
        for v in vals:
            axs[0,0].text(v[0], v[1], f'{int(v[1])}', ha='center', va='bottom', fontsize=9)
        axs[0,0].set_title('Effect of Population Size')
        axs[0,0].set_xlabel('Population Size')
        axs[0,0].set_ylabel('Best Fitness')
        
    # 2. Crossover Probability Chart (Top-Right)
    if 'crossover' in results_dict:
        vals = results_dict['crossover']
        # Plot probability vs score
        axs[0,1].plot([v[0] for v in vals], [v[1] for v in vals], marker='o', color='green')
        for v in vals:
            axs[0,1].text(v[0], v[1], f'{int(v[1])}', ha='center', va='bottom', fontsize=9)
        axs[0,1].set_title('Effect of Crossover Probability')
        axs[0,1].set_xlabel('Crossover Probability (Px)')
        axs[0,1].set_ylabel('Best Fitness')
        
    # 3. Mutation Probability Chart (Bottom-Left)
    if 'mutation' in results_dict:
        vals = results_dict['mutation']
        # Plot probability vs score
        axs[1,0].plot([v[0] for v in vals], [v[1] for v in vals], marker='o', color='magenta')
        for v in vals:
            axs[1,0].text(v[0], v[1], f'{int(v[1])}', ha='center', va='bottom', fontsize=9)
        axs[1,0].set_title('Effect of Mutation Probability')
        axs[1,0].set_xlabel('Mutation Probability (Pm)')
        axs[1,0].set_ylabel('Best Fitness')
        
    # 4. Mutation Type Chart (Bottom-Right, Bar Chart)
    if 'mutation_type' in results_dict:
        vals = results_dict['mutation_type']
        types = [v[0] for v in vals] # Get types (swap, inversion)
        scores = [v[1] for v in vals] # Get scores
        # Create bar chart
        axs[1,1].bar(types, scores, color=['steelblue','orange'])
        # Add text labels
        for i, v in enumerate(scores):
            axs[1,1].text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        axs[1,1].set_title('Effect of Mutation Type')
        axs[1,1].set_xlabel('Mutation Type')
        axs[1,1].set_ylabel('Best Fitness')

    # Add optimal line to all charts for reference
    optimal_dict = {'Berlin11': 4038, 'Berlin52': 7542, 'kroA100': 21282, 'kroA150': 26524}
    for ax in axs.flat: # Loop through all 4 subplots
        if instance_name in optimal_dict:
            # Draw dashed red line at optimal value
            ax.axhline(optimal_dict[instance_name], color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.2) # Add grid
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout leaving space for title
    plt.savefig(f'results/{instance_name}_param_comparison.png') # Save figure
    plt.close() # Close figure

def run_param_comparison(cities_df, instance_name):
    """
    Runs the GA multiple times with different settings (parameters)
    to find out which settings work best.
    Returns a dictionary of results.
    """
    print(f"\nRunning parameter comparison for {instance_name}...")
    results = {} # Dictionary to store results
    
    # Test 1: Population Sizes
    pop_sizes = [30, 50, 100, 150]
    pop_results = []
    print("  Testing Population Sizes...")
    for pop_size in pop_sizes:
        # Run GA with reduced epochs for speed
        _, best_score, _ = run_genetic_algorithm(cities_df, epochs=50, pop_size=pop_size, mutation_type='swap') 
        pop_results.append((pop_size, best_score))
    results['pop_size'] = pop_results
    
    # Test 2: Crossover Probabilities
    cross_probs = [0.6, 0.8, 1.0]
    cross_results = []
    print("  Testing Crossover Probabilities...")
    for px in cross_probs:
        # Run GA with specific crossover probability
        _, best_score, _ = run_genetic_algorithm(cities_df, epochs=50, pop_size=100, crossover_prob=px, mutation_type='swap')
        cross_results.append((px, best_score))
    results['crossover'] = cross_results
    
    # Test 3: Mutation Probabilities
    mut_probs = [0.01, 0.1, 0.2]
    mut_results = []
    print("  Testing Mutation Probabilities...")
    for pm in mut_probs:
        # Run GA with specific mutation probability
        _, best_score, _ = run_genetic_algorithm(cities_df, epochs=50, pop_size=100, mutation_prob=pm, mutation_type='swap')
        mut_results.append((pm, best_score))
    results['mutation'] = mut_results
    
    # Test 4: Mutation Type (Swap vs Inversion)
    print("  Testing Mutation Types (Swap vs Inversion)...")
    # Run with Swap mutation
    _, best_swap, _ = run_genetic_algorithm(cities_df, epochs=100, pop_size=100, mutation_prob=0.1, mutation_type='swap')
    # Run with Inversion mutation
    _, best_inversion, _ = run_genetic_algorithm(cities_df, epochs=100, pop_size=100, mutation_prob=0.1, mutation_type='inversion')
    
    # Store results
    results['mutation_type'] = [('swap', best_swap), ('inversion', best_inversion)]
    
    # Create the comparison chart
    plot_param_comparison(results, instance_name)
    return results

def main():
    """
    Main function.
    Orchestrates the whole process:
    1. Setup folders
    2. Loop through files
    3. Run all algorithms (Greedy, Random, GA)
    4. Generate charts
    """
    # Define the directory name for results
    results_dir = 'results'
    # Check if the directory already exists
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir) # Clear old results by deleting the directory
    os.makedirs(results_dir, exist_ok=True) # Create a fresh new directory

    # List of files to process with their names and optimal scores
    instances = [
        ("berlin 11 tsp.txt", "Berlin11", 4038),
        ("berlin52 tsp.txt", "Berlin52", 7542),
        ("kroA100 tsp.txt", "kroA100", 21282),
        ("kroA150 tsp.txt", "kroA150", 26524)
    ]
    
    # Loop through each instance in the list
    for filename, instance_name, optimal in instances:
        # Check if the file exists in the current directory
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.") # Print error if missing
            continue # Skip to next file
            
        print(f"\n{'='*60}")
        print(f"Processing: {instance_name}") # Print header
        print(f"{'='*60}")
        
        # 1. Parse Data
        print("\n1. Parsing TSP file...")
        # Call the parser function
        coords, city_ids, id_to_index = parse_tsp_file(filename)
        # Bundle data into a tuple for easy passing
        cities_df = (coords, city_ids, id_to_index)
        print(f"   Loaded {len(city_ids)} cities")
        
        # 2. Run Greedy Algorithm
        print("\n2. Running greedy algorithm...")
        best_greedy_score = float('inf') # Initialize best score
        # Try starting from every city to find best greedy path
        for start_city in city_ids:
            solution = greedy_algorithm(cities_df, start_city) # Run greedy
            score = calculate_fitness(solution, cities_df) # Calculate score
            if score < best_greedy_score:
                best_greedy_score = score # Update best score
        print(f"   Best greedy score: {best_greedy_score:.2f}")
        
        # 3. Run Random Search (for baseline comparison)
        print("\n3. Generating random solutions...")
        random_scores = []
        for _ in range(100): # Generate 100 random solutions
            random_sol = create_random_solution(city_ids)
            score = calculate_fitness(random_sol, cities_df)
            random_scores.append(score)
        random_best = min(random_scores) # Find best random score
        random_avg = np.mean(random_scores) # Calculate average random score
        
        # 4. Run Genetic Algorithm (The main event)
        # Using Inversion mutation as it typically performs better for TSP
        print("\n4. Running Genetic Algorithm (Main Run)...")
        best_solution, best_score, fitness_history = run_genetic_algorithm(
            cities_df, epochs=200, pop_size=100, mutation_type='inversion'
        )
        
        # 5. Create Visualization Charts
        print("\n5. Creating visualization...")
        # Plot progress graph
        plot_results(fitness_history, instance_name)
        # Plot comparison bar chart
        plot_comparison(optimal, best_score, best_greedy_score, random_best, random_avg, instance_name)
        # Plot the best tour map
        plot_tour(best_solution, cities_df, best_score, instance_name)
        
        # 6. Run Parameter Comparison (Optional deep dive)
        print("\n6. Parameter comparison...")
        run_param_comparison(cities_df, instance_name)
        
        print(f"\nFinished processing {instance_name}!")


if __name__ == "__main__":
    main()