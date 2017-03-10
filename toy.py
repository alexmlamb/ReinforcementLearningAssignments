import numpy.random as rng


states = ['a', 'b', 'c']

mu = {}

mu[('a','c')] = 0.01
mu[('a','b')] = 0.99
mu[('a','a')] = 0.0

mu[('b','a')] = 0.2
mu[('b','b')] = 0.3
mu[('b','c')] = 0.5

mu[('c','a')] = 0.2
mu[('c','b')] = 0.4
mu[('c','c')] = 0.4

pi = {}

pi[('a','c')] = 1.0
pi[('a','b')] = 0.0
pi[('a','a')] = 0.0

pi[('b','a')] = 0.0
pi[('b','b')] = 0.0
pi[('b','c')] = 1.0

pi[('c','a')] = 1.0
pi[('c','b')] = 0.0
pi[('c','c')] = 0.0

rewards = {}

rewards['a'] = 0.0
rewards['b'] = -2.0
rewards['c'] = 10.0

'''
    Pick the next state with a probability proportional to the policy
'''
def sample(policy,start_state):
    probs = []

    for state in states:
        probs.append(policy[(start_state,state)])

    assert sum(probs) == 1.0

    samp = rng.multinomial(n=1, pvals=probs).argmax()

    return states[samp]

def evaluate_policy(policy,start_state,num_steps):
    reward_sum = 0.0
    state = start_state
    for i in range(0, num_steps):
        reward_sum += rewards[state]
        state = sample(policy, state)

    return reward_sum

def importance_evaluate_policy(behavior_policy, target_policy, start_state, num_steps):
    pass

def retrace_evaluate_policy():
    pass

if __name__ == "__main__":

    '''
        Evaluating with the actual target policy (just to get ground truth for evaluation)
    '''
    
    true_reward = evaluate_policy(pi, 'a', 10)
    
    print "true reward under target policy", true_reward

    '''
        Evaluating the target policy with an importance sampling estimator
    '''



