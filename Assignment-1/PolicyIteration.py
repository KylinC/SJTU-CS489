import numpy as np
from gridworld import GridworldEnv

env = GridworldEnv([6,6])

def policy_iteration(env, theta=0.001, discount_factor=1.0):
    """
    Policy Iteration Algorithm.
    
    Args:
        env: gridWorld
        theta: Stopping threshold. 
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    def one_step_lookahead(state, V):

        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """

        A = 0.0
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A += 0.25 * (reward + discount_factor * V[next_state])
        return A

    def greedy_policy_choose(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] = V[next_state]
        return np.argmax(A)
    
    """
    Policy Iteration
    """

    V = np.zeros(env.nS)
    Vtmp = np.zeros(env.nS)
    iteration_step = 0
    while True:
        iteration_step += 1
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Calculate the new value
            new_action_value = one_step_lookahead(s, V)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(new_action_value - V[s]))
            # Update the value function
            Vtmp[s] = new_action_value        
        # Check if we can stop 
        if delta < theta:
            print("iterations:",iteration_step)
            break
        else:
            V = Vtmp
    
    """
    Policy Evaluation 
    """

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        best_action = greedy_policy_choose(s, V)
        # Always take the best action
        policy[s, best_action] = 1.0
    return policy, V

policy, v = policy_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=n, 1=e, 2=s, 3=w):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Final state:")
print(np.reshape(v, env.shape))
print("")