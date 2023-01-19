# Dyna-Q is a MODEL-BASED reinforcement learning method 
import os, sys, numpy, random
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld

def epsilon_greedy(q, state, epsilon):
    if numpy.random.random() < epsilon:
        return numpy.random.choice(q.shape[1])
    return q[state].argmax()

def dynaQ( environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99 ):

    Q = numpy.zeros((environment.observation_space, environment.action_space))
    M = numpy.array([[[None, None] for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
    memory = []

    # Not Random Version
    # s = environment.random_initial_state()

    #while True: # In teoria 
    for _ in range(maxiters):
        # (A)
        s = environment.random_initial_state()
        # (B)
        a = epsilon_greedy(Q,s,eps) 
        # (C)
        new_state = environment.sample(a,s) 
        reward = environment.R[new_state]  
        visited = list([s,a])
        if visited not in memory:
            print(f"Added {visited}")
            memory.append(visited) # save for selecting randomly next 

        # (D)
        val = [0 for _ in environment.actions]
        for action in environment.actions: 
            val[action]= Q[new_state,action]
        Q[s,a] = Q[s,a] + alfa*(reward + gamma * max(val) - Q[s,a])
        # (E)
        M[s,a] = [reward,new_state]
        # (F)
        for _ in range(n):

            # Scelta Randomica s e a 
            index = random.choice(range(len(memory)))
            s,a = memory[index]

            reward, new_state_f = M[s,a]   
            val = [0 for _ in environment.actions]
            for action in environment.actions: 
                val[action]= Q[new_state_f,action]
            Q[s,a] = Q[s,a] + alfa*(reward + gamma * max(val) - Q[s,a])

        # Not Random Version
        # s = new_state

    
    policy = Q.argmax(axis=1) 
    return policy

def main():
    print( "\n************************************************" )
    print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
    print( "*                  (Dyna-Q)                      *" )
    print( "**************************************************" )

    print("\nEnvironment Render:")
    env = GridWorld( deterministic=True )
    env.render()

    print( "\n6) Dyna-Q" )
    dq_policy_n00 = dynaQ( env, n=0  )
    dq_policy_n25 = dynaQ( env, n=25 )
    dq_policy_n50 = dynaQ( env, n=50 )

    env.render_policy( dq_policy_n50 )
    print()
    print( f"\tExpected reward with n=0 :", env.evaluate_policy(dq_policy_n00) )
    print( f"\tExpected reward with n=25:", env.evaluate_policy(dq_policy_n25) )
    print( f"\tExpected reward with n=50:", env.evaluate_policy(dq_policy_n50) )

if __name__ == "__main__":
    main()
