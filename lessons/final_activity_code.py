import warnings; warnings.filterwarnings("ignore")
import sys, os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from point_discrete import PointNavigationDiscrete
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

def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ): 
    """
    ###Notes###
    One NN for The Value,state function and one for the policy prediction 
    
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
    memory_buffer = np.array(memory_buffer)
    for _ in range(10):
        np.random.shuffle(memory_buffer)
        states = np.array(list(memory_buffer[:, 0]), dtype=np.float)[:,0,:]
        rewards = np.array(list(memory_buffer[:, 2]), dtype=np.float)
        next_states = np.array(list(memory_buffer[:, 3]), dtype=np.float)[:,0,:]
        done = np.array(list(memory_buffer[:, 4]), dtype=bool)
        with tf.GradientTape() as critic_tape:
            target = rewards + (1 - done.astype(int)) * gamma * critic_net(next_states).numpy()
            prediction = critic_net(states)
            objective = tf.math.square(prediction - target)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))

    objectives = []
    with tf.GradientTape() as actor_tape:
        actions = np.array(list(memory_buffer[:, 1]), dtype=int)
        adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
        adv_b = critic_net(states).numpy().reshape(-1)
        probs = actor_net(states)
        indices = tf.transpose(tf.stack([tf.range(probs.shape[0]), actions]))
        probs = tf.gather_nd(
            indices=indices,
            params=probs
        )
        objective = tf.math.log(probs) * (adv_a - adv_b)
        objectives.append(tf.reduce_mean(tf.reduce_sum(objective)))
        objective = - tf.math.reduce_mean(objectives)
        grads = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))



def training_loop( env, actor_net, critic_net, updateRule, frequency=4, episodes=100 ): 
    actor_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    critic_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    success_list, success_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    for ep in range(episodes):
    
        state = env.reset()[0] 
        state = state.reshape(-1,9)
        ep_reward,ep_lenght= 0,0
        # info = {} 
        while True:

            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n,p=distribution)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,9)
            memory_buffer.append([state,action,reward,next_state,terminated])
            ep_reward += reward
            ep_lenght+= 1
            # max_state = max(max_state,next_state[0][0])
            if terminated or truncated:  break
            state = next_state
        
        # Perform the training every 'frequency'(10) episodes
        # memory_buffer.append(partial_memory_buffer)
        if ep %frequency == 0 and ep!=0: 
            updateRule( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer) # critic_memory_buffer
            memory_buffer = []
    
        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        success_queue.append( info["goal_reached"] )
        success_list.append( np.mean(success_queue) )
        print( f"episode {ep:4d}: reward: {ep_reward:5.2f} (averaged: {np.mean(reward_queue):5.2f}), success rate ({int(np.mean(success_queue)*100):3d}/100) lenght: {ep_lenght}" )

    #Close the enviornment and return the rewards list
    env.close()
    return success_list 


class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step( action )
        # Extract the information from the observations
        old_heading, old_distance, old_lidars = self.previous_observation[0], self.previous_observation[1], self.previous_observation[2:]
        heading, distance, lidars = observation[0], observation[1], observation[2:]
        
        # Exploting useful flags
        goal_reached = bool(info["goal_reached"])
        collision = bool(info["collision"])

        # Distance
        # reward = (distance - old_distance)*10
        # Lidars 
        # positions = []
        # for i,lid in enumerate(lidars):
        #     if lid<=0.5: positions.append(i) 
        # for val in positions:
        #     reward+= -(lidars[val]-old_lidars[val])*10

        # Headings 
        reward = -1
        if action == 0  and (-0.1<heading< 0.1): reward = 1
        if action == 1  and (heading < -0.1): reward = 1
        if action == 2  and (heading > 0.1): reward = 1
        if goal_reached: reward = 100
        if collision: reward = -10
        return observation, reward, terminated, truncated, info
	

def main(): 
    print( "\n*****************************************************" )
    print( "*    Welcome to the final activity of the RL-Lab    *" )
    print( "*                                                   *" )
    print( "*****************************************************\n" )
    
    _training_steps = 1000
    	
    # Load the environment and override the reward function
    env = PointNavigationDiscrete( ) #optional: render_mode="human"
    env = OverrideReward(env)
    
    # Create the networks and perform the actual training
    n_state,n_actions = 9,3
    actor_net = createDNN( n_state, n_actions, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN(n_state, 1, nLayer=2, nNodes=32, last_activation="linear")
    success_training = training_loop( env, actor_net, critic_net, A2C , frequency=5, episodes=_training_steps  )
    
    # Save the trained neural network
    actor_net.save( "490856_mangrella.h5" )
    
    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, success_training, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "success", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()	
