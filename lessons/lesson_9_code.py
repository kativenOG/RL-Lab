import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


def createDNN( nInputs, nOutputs, nLayer, nNodes ): 
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) #input layer + hidden layer #1
    for _ in range(1,nLayer):
        model.add(Dense(nNodes, activation="relu")) #hidden layer #2
    model.add(Dense(nOutputs, activation="softmax")) #output layer with Softmax, 
    # Our Network rappresented a policy and not the q function compared to DEEPQL, so values have to be between 0 and 1 
    
    return model


def training_loop( env, neural_net, updateRule, frequency=10, episodes=100 ):
    """
    Main loop of the reinforcement learning algorithm. Execute the actions and interact
    with the environment to collect the experience for the training.
    
    Args:
    	env: gymnasium environment for the training
    	neural_net: the model to train 
    	updateRule: external function for the training of the neural network
    	
    Returns:
    	averaged_rewards: array with the averaged rewards obtained
    
    """
    
    optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer,memory_buffer_partial = [],[]
    for ep in range(episodes):
    
        state = env.reset()[0] 
        state = state.reshape(-1,4)
        ep_reward = 0
        while True:
        
            action = env.action_space.sample() 
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,4)
            
            memory_buffer_partial.append( list([state,action,reward,next_state,terminated])) 
            ep_reward += reward
            
            if terminated or truncated:  break
            state = next_state
        
        #TODO: Perform the actual training every 'frequency' episodes
        memory_buffer.append(np.array(memory_buffer_partial)) # Cast to np Array for Slicing  
        memory_buffer_partial = []
        if ep %frequency == 0: 
            updateRule( neural_net, np.array(memory_buffer), optimizer )
            memory_buffer = []

        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )
    
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def REINFORCE_naive( neural_net, memory_buffer, optimizer ):
    """
    Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.
    
    """

    # TAPE SETUP 
    objectives= []
    with tf.GradientTape() as tape:

        for index in range(len(memory_buffer)):
            # state, action, reward, next_state, done = memory_buffer[][index] 
            state = np.array(memory_buffer[index][:,0])
            action = memory_buffer[index][:,1]
            reward = memory_buffer[index][:,2]
            
            target,rewards = 0,0
            for i in range(len(state)):
                rewards+= reward[i]
                appo = neural_net(state[i])
                appo = appo[0][action[i]]
                target += tf.math.log(appo)
            target = target * rewards
            objectives.append(target)
        
        objective= - tf.math.reduce_mean(objectives)
        grad = tape.gradient(objective,neural_net.trainable_variables,)
        optimizer.apply_gradients( zip(grad, neural_net.trainable_variables) )


def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
    """
    Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,
    (same as REINFORCE_naive but with rw2go Update Rule )    
    """
    # TAPE SETUP 
    objectives= []
    with tf.GradientTape() as tape:

        for index in range(len(memory_buffer)):
            state = np.array(memory_buffer[index][:,0])
            action = memory_buffer[index][:,1]
            reward = memory_buffer[index][:,2]
            
            target,rewards = 0,0
            for i in range(len(state)):
                rewards = sum(reward[i:])
                appo = neural_net(state[i])
                appo= appo[0][action[i]]
                target += tf.math.log(appo)*rewards
            objectives.append(target)
        
        objective= - tf.math.reduce_mean(objectives)
        grad = tape.gradient(objective,neural_net.trainable_variables,)
        optimizer.apply_gradients( zip(grad, neural_net.trainable_variables) )
    


def main():
    print( "\n*************************************************" )
    print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
    print( "*                 (REINFORCE)                   *" )
    print( "*************************************************\n" )
    
    _training_steps = 1500
    env = gymnasium.make( "CartPole-v1" )
    
    # Training A)
    neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
    rewards_naive = training_loop( env, neural_net, REINFORCE_naive, episodes=_training_steps  )
    print()
    
    # Training B)
    neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
    rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go, episodes=_training_steps  )
    
    # Plot
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="naive", linewidth=3)
    plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()	
