import numpy as np

def monte_carlo_paths(S_0, time_to_expiry, sigma, drift, seed, n_sims, n_timesteps):
    """
    Create random paths of a underlying following a browian geometric motion
    
    input:
    
    S_0 = Spot at t_0
    time_to_experiy = end of the timeseries (last observed time)
    sigma = the volatiltiy (sigma in the geometric brownian motion)
    drift = drift of the process
    n_sims = number of paths to generate
    n_timesteps = numbers of aquidistant time steps 
    
    return:
    
    a (n_timesteps x n_sims x 1) matrix
    """
    if seed > 0:
            np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_sims, n_timesteps)
    S = S_0
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    r = drift
    # See Advanced Monte Carlo methods.py for barrier and related exotic options by Emmanuel Gobet
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis=1)
    return np.reshape(np.transpose(np.c_[np.ones(n_sims)*S_0, S_T]), (n_timesteps+1, n_sims, 1))



def compute_avg_return(environment, policy, num_episodes):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset() # timestamp 형식 반환
    episode_return = 0.0
    print("-------------------iter start---------------")
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      print(f'validate : {action_step.action}')
      episode_return += time_step.reward

      #print(episode_return)
    print("----------------------------")
    total_return += episode_return
  avg_return = total_return / num_episodes
  #if avg_return < 6000 : print("avg_return : {}".format(avg_return))
  return avg_return.numpy()[0]
