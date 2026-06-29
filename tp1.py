import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True 
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human' #selecione esta opção para visualizar o ambiente (testes mais lentos)
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos) 
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


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
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        print("Aterragem bem sucedida!")
        return True

    print("Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success


# --- PERCEPTIONS ---
def is_touching_ground(observation):
    return observation[6] == 1 or observation[7] == 1

def is_flipping_left(observation):
    return (observation[4] + observation[5]) > 0.3

def is_flipping_right(observation):
    return (observation[4] + observation[5]) < -0.3

def is_plummeting(observation):
    y, vy = observation[1], observation[3]
    if y < 0.3:
        return vy < -0.2
    return vy < -0.5

def is_falling_fast(observation):
    y = observation[1]
    vy = observation[3]

    if y > 0.5:
        return vy < -0.1
    else:
        return vy < -0.20

def is_drifting_right(observation):
    return (observation[0] + 2.0 * observation[2]) > 0.05

def is_drifting_left(observation):
    return (observation[0] + 2.0 * observation[2]) < -0.05


# --- ACTIONS ---
def do_nothing(): return np.array([0.0, 0.0])

def fire_main_full(): return np.array([1.0, 0.0])
def fire_main_gentle(): return np.array([0.75, 0.0]) 

def fix_flip_right(): return np.array([0.0, 1.0]) 
def fix_flip_left(): return np.array([0.0, -1.0]) 

def steer_left(): return np.array([0.0, -1.0])
def steer_right(): return np.array([0.0, 1.0])


# --- AGENT ---
def reactive_agent(observation):

    if is_touching_ground(observation): return do_nothing()
        
    elif is_flipping_left(observation): return fix_flip_right()
    elif is_flipping_right(observation): return fix_flip_left()
     
    elif is_plummeting(observation): return fire_main_full()            
   
    elif is_drifting_right(observation): return steer_left()
    elif is_drifting_left(observation): return steer_right()
    
    elif is_falling_fast(observation): return fire_main_gentle()
        
    else: return do_nothing()
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    
#our main loop
success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)
    

