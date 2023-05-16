# DISCRETO, non va
# from lesson_10_code import A2C
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): 
    # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) #input layer + hidden layer #1
    for _ in range(1,nLayer):
        model.add(Dense(nNodes, activation="relu")) #hidden layer 
    model.add(Dense(nOutputs, activation=last_activation)) #output layer
    return model

def training_loop( env, actor_net, critic_net, updateRule, frequency=5, episodes=100 ): 
    actor_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    critic_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    for ep in range(episodes):
    
        state = np.array(env.reset()[0]).reshape(-1,1)
        ep_reward,ep_lenght= 0,0
        
        while True:


            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n,p=distribution)
            # action = actor_net(state).numpy().tolist() 
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state).reshape(-1,1)
            memory_buffer.append([state,action,reward,next_state,terminated])
            ep_reward += reward
            ep_lenght+= 1
            if terminated or truncated:  
                if terminated and state == 15: print("GOAL")
                break
            state = next_state
        
        if ep %frequency == 0 and ep!=0: 
            updateRule( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer) # critic_memory_buffer
            memory_buffer = []
    
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} Best_Distance: (averaged: {np.mean(reward_queue):5.2f}) len: {ep_lenght} ")  #
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list

	
def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    ###Notes###
    One NN for The Value,state function and one for the policy prediction 
    
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
    memory_buffer = np.array(memory_buffer)
    states,rewards,next_states = 0,0,0
    for _ in range(10):
        np.random.shuffle(memory_buffer)
        states = np.array(list(memory_buffer[:, 0]), dtype=np.int32)
        rewards = np.array(list(memory_buffer[:, 2]), dtype=np.float32)
        next_states = np.array(list(memory_buffer[:, 3]), dtype=np.int32)
        done = np.array(list(memory_buffer[:, 4]), dtype=bool)
        with tf.GradientTape() as critic_tape:
            target = rewards + (1 - done.astype(int)) * gamma * critic_net(next_states).numpy()
            prediction = critic_net(states)
            objective = tf.math.square(prediction - target)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))

    with tf.GradientTape() as actor_tape:
        actions = np.array(list(memory_buffer[:, 1]), dtype=np.int32)
        adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
        adv_b = critic_net(states).numpy().reshape(-1)
        probs = actor_net(states)
        indices = tf.transpose(tf.stack([tf.range(probs.shape[0]), actions]))
        probs = tf.gather_nd(
            indices=indices,
            params=probs
        )
        objective =  tf.math.log(probs) * (adv_a - adv_b)
        objectives = - tf.reduce_mean(tf.reduce_sum(objective))
        grads = actor_tape.gradient(objectives, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))


class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment
    """
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        max_state = 15 # 3*4 + 3 
        # previous_state = self.env.nrow*4 + self.env.ncol -1
        # new_reward = observation - previous_state # * 10e-1
        reward = - (15 - observation) *10e-1
        # if terminated and (observation != max_state): reward = -100
        if observation == max_state: reward = 100
        elif terminated: reward = -10
        return observation, reward, terminated, truncated, info
        
def main(): 
    print( "\n***************************************************" )
    print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
    print( "*                 (DRL in Practice)               *" )
    print( "***************************************************\n" )

    _training_steps = 2000
    # Crete the environment and add the wrapper for the custom reward function
    gymnasium.envs.register( id='FrozenLake-v1', entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',max_episode_steps=1000)
    env = gymnasium.make("FrozenLake-v1",map_name="4x4",is_slippery=False) #,render_mode="human")
    env = OverrideReward(env)
    # Actor  = n_stato, n_azioni
    # Critic = n_stato, 1 
    actor_net =  createDNN(1,4, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN(1,1, nLayer=2, nNodes=32, last_activation="linear") 
    rewards_training = training_loop(env, actor_net, critic_net, A2C, frequency=2, episodes=_training_steps  )

    # Save the trained neural network
    actor_net.save( "MountainCarActor.h5" )



if __name__ == "__main__":
	main()	
