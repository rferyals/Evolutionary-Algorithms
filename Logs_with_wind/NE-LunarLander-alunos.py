import random
import copy
import sys
import numpy as np
import gymnasium as gym
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = None
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]



PROB_MUTATION = 0.008
PROB_CROSSOVER = 0.5
ELITE_SIZE = 0
POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100

STD_DEV = 0.1

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)

    if legs_touching and on_landing_pad and stable_velocity and stable_orientation:
        return True
    return False

def objective_function(observation_history):
    obs = observation_history[-1]

    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    theta, omega = obs[4], obs[5]
    leg_l, leg_r = obs[6], obs[7]

    fitness = 0.0
    
    # Penalize distance from the landing zone center.
    fitness -= 3.0 * abs(x)
    fitness -= 2.0 * abs(y)
    
    # Penalize unsafe motion and rotation.
    fitness -= 3.0 * abs(vx)
    fitness -= 6.0 * abs(vy)
    fitness -= 2.0 * abs(theta)
    fitness -= 1.0 * abs(omega)

    # Heavy penalty for hard contact with any leg.
    if (leg_l or leg_r) and vy < -0.5:
        fitness -= 200.0
    # Small reward for soft leg contact.
    elif leg_l or leg_r:
        fitness += 5.0 * leg_l + 5.0 * leg_r

    # Check the last part of the flight near the landing pad.
    safe_zone_steps = 0
    safe_zone_penalty = 0.0
    for h in observation_history:
        hx, hy = h[0], h[1]
        hvx, hvy = h[2], h[3]
        htheta, homega = h[4], h[5]

        in_pad = abs(hx) <= 0.2
        near_ground = hy <= 0.35
        if in_pad and near_ground:
            safe_zone_steps += 1

            # Reward calm motion near the pad.
            if hvy > -0.05:
                safe_zone_penalty += 1.5 * (hvy + 0.05)

            safe_zone_penalty += 0.8 * abs(hvx)
            safe_zone_penalty += 0.8 * abs(homega)
            safe_zone_penalty += 0.5 * abs(htheta)

    if safe_zone_steps > 0:
        fitness -= safe_zone_penalty / safe_zone_steps
        fitness += 8.0

    # Give a bonus for a successful landing.
    success = check_successful_landing(obs)
    if success:
        fitness += 120.0

    return fitness, success

def simulate(genotype, render_mode=None, seed=None, env=None,
             enable_wind=None, wind_power=None, turbulence_power=None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    # Wind params fall back to module-level globals when not provided explicitly
    if enable_wind       is None: enable_wind       = ENABLE_WIND
    if wind_power        is None: wind_power        = WIND_POWER
    if turbulence_power  is None: turbulence_power  = TURBULENCE_POWER

    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode=render_mode,
        continuous=True, gravity=GRAVITY,
        enable_wind=enable_wind, wind_power=wind_power,
        turbulence_power=turbulence_power)

    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue,
             enable_wind=False, wind_power=15.0, turbulence_power=0.0):
    #Evaluates individuals until it receives None.
    #Runs on a worker process — wind params MUST be passed explicitly
    #because spawned processes cannot see globals updated in the parent.
    env = gym.make("LunarLander-v3", render_mode=None,
        continuous=True, gravity=GRAVITY,
        enable_wind=enable_wind, wind_power=wind_power,
        turbulence_power=turbulence_power)
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind['fitness'] = simulate(ind['genotype'], seed=None, env=env)[0]
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    # Pick 5 random individuals for a tournament.
    tournament = random.sample(population, 5)
    # Return a copy of the best individual by fitness.
    return copy.deepcopy(max(tournament, key=lambda ind: ind['fitness']))

def crossover(p1, p2):
    # Read parent genotypes.
    g1, g2 = p1['genotype'], p2['genotype']
    # Build a child by mixing genes from both parents.
    child = []
    for x, y in zip(g1, g2):
        # For each gene, randomly choose parent 1 or parent 2.
        if random.random() < 0.5:
            child.append(x)
        else:
            child.append(y)
    # New child has no evaluated fitness yet.
    return {'genotype': child, 'fitness': None}

def mutation(p):
    # Access genotype and current fitness.
    genotype = p['genotype']
    fit = p.get('fitness')

    # Use stronger mutation when fitness is poor.
    if fit is None or fit < -30:
        std = 0.8
        prob = PROB_MUTATION * 3
    elif fit < 20:
        std = 0.3
        prob = PROB_MUTATION * 2
    else:
        std = 0.1
        prob = PROB_MUTATION

    # Mutate each gene with Gaussian noise.
    for i in range(len(genotype)):
        if random.random() < prob:
            genotype[i] += random.gauss(0, std)

    # Return the mutated individual.
    return p
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution(enable_wind=False, wind_power=15.0, turbulence_power=0.0):
    #Create evaluation processes — wind params forwarded explicitly to workers
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(
            target=evaluate,
            args=(evaluationQueue, evaluatedQueue,
                  enable_wind, wind_power, turbulence_power)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append((eval(fitness), eval(shape), eval(genotype)))
    return bests

def load_config(config_file):
    """Parse experiments_config.txt into a list of experiment dicts.
    Supports 7-column (no wind) and 10-column (with wind) formats."""
    experiments = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            experiments.append({
                'id':               int(parts[0]),
                'mutation_prob':    float(parts[1]),
                'crossover_prob':   float(parts[2]),
                'elite_size':       int(parts[3]),
                'pop_size':         int(parts[4]),
                'generations':      int(parts[5]),
                'n_runs':           int(parts[6]),
                # Wind columns are optional — default to no wind
                'wind':             int(parts[7])   if len(parts) > 7 else 0,
                'wind_power':       float(parts[8]) if len(parts) > 8 else 15.0,
                'turbulence_power': float(parts[9]) if len(parts) > 9 else 0.0,
            })
    return experiments

def run_experiments(config_file):
    """Run every experiment defined in config_file and save per-run log files."""
    global PROB_MUTATION, PROB_CROSSOVER, ELITE_SIZE, POPULATION_SIZE, \
           NUMBER_OF_GENERATIONS, ENABLE_WIND, WIND_POWER, TURBULENCE_POWER

    experiments = load_config(config_file)
    seeds = [140, 751, 513, 913, 489]

    os.makedirs('results', exist_ok=True)

    for exp in experiments:
        exp_id                = exp['id']
        PROB_MUTATION         = exp['mutation_prob']
        PROB_CROSSOVER        = exp['crossover_prob']
        ELITE_SIZE            = exp['elite_size']
        POPULATION_SIZE       = exp['pop_size']
        NUMBER_OF_GENERATIONS = exp['generations']
        n_runs                = exp['n_runs']
        ENABLE_WIND           = bool(exp['wind'])
        WIND_POWER            = exp['wind_power']
        TURBULENCE_POWER      = exp['turbulence_power']

        exp_dir = os.path.join('results', f'exp{exp_id}')
        os.makedirs(exp_dir, exist_ok=True)

        # Save metadata so test mode can restore the correct wind settings
        with open(os.path.join(exp_dir, 'meta.txt'), 'w') as mf:
            mf.write(f'wind={int(ENABLE_WIND)}\n')
            mf.write(f'wind_power={WIND_POWER}\n')
            mf.write(f'turbulence_power={TURBULENCE_POWER}\n')

        wind_label = (f'WIND ON  power={WIND_POWER}  turb={TURBULENCE_POWER}'
                      if ENABLE_WIND else 'no wind')
        print(f"\n{'='*65}")
        print(f"Experiment {exp_id}  |  mut={PROB_MUTATION}  cross={PROB_CROSSOVER}"
              f"  elite={ELITE_SIZE}  pop={POPULATION_SIZE}  gens={NUMBER_OF_GENERATIONS}")
        print(f"             |  {wind_label}")
        print(f"{'='*65}")

        for run in range(n_runs):
            print(f"\n--- Exp {exp_id} | Run {run+1}/{n_runs} (seed={seeds[run]}) ---")
            random.seed(seeds[run])
            bests = evolution(
                enable_wind=ENABLE_WIND,
                wind_power=WIND_POWER,
                turbulence_power=TURBULENCE_POWER)

            log_path = os.path.join(exp_dir, f'log{run}.txt')
            with open(log_path, 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
            print(f"Saved {log_path}")

        print(f"\nExperiment {exp_id} complete.")

if __name__ == '__main__':

    evolve_or_test = int(sys.argv[1])

    if evolve_or_test == 0:
        #--to evolve the controller--
        evolve = True
        render_mode = None
    elif evolve_or_test == 1:
        #--to test the evolved controller without visualisation--
        evolve = False
        render_mode = None
    elif evolve_or_test == 2:
        #--to test the evolved controller with visualisation--
        evolve = False
        render_mode = 'human'


    if evolve:
        # Run all experiments from config file:
        #   python NE-LunarLander-alunos.py 0 experiments_config.txt
        if len(sys.argv) > 2:
            run_experiments(sys.argv[2])
        else:
            # Legacy single-run using current global parameters
            seeds = [140, 751, 513, 913, 489]
            os.makedirs('results/single', exist_ok=True)
            for i in range(5):
                random.seed(seeds[i])
                bests = evolution()
                log_path = f'results/single/log{i}.txt'
                with open(log_path, 'w') as f:
                    for b in bests:
                        f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')
                print(f"Saved {log_path}")

    else:
        # Test an evolved individual:
        #   python NE-LunarLander-alunos.py 1 results/exp1/log0.txt
        #   python NE-LunarLander-alunos.py 2 results/exp9/log0.txt
        filename = sys.argv[2] if len(sys.argv) > 2 else 'results/single/log0.txt'
        bests = load_bests(filename)
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]

        # Auto-detect wind settings from meta.txt saved next to the log file
        test_wind        = False
        test_wind_power  = 15.0
        test_turb        = 0.0
        meta_path = os.path.join(os.path.dirname(filename), 'meta.txt')
        if os.path.isfile(meta_path):
            with open(meta_path) as mf:
                for mline in mf:
                    key, val = mline.strip().split('=')
                    if key == 'wind':             test_wind       = bool(int(val))
                    elif key == 'wind_power':     test_wind_power = float(val)
                    elif key == 'turbulence_power': test_turb     = float(val)
            print(f'Wind settings from meta.txt: wind={test_wind}  '
                  f'power={test_wind_power}  turb={test_turb}')
        else:
            print('No meta.txt found — testing with no wind (default).')

        ntests = TEST_EPISODES
        fit, success = 0, 0
        for i in range(1, ntests + 1):
            f, s = simulate(ind, render_mode=render_mode, seed=None,
                            enable_wind=test_wind,
                            wind_power=test_wind_power,
                            turbulence_power=test_turb)
            fit += f
            success += s
        print(f'Average fitness: {fit/ntests:.3f}   Success rate: {success/ntests:.3f}')
