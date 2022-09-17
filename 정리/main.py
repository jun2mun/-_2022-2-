from utils.D_H_math import monte_carlo_paths
import matplotlib.pyplot as plt

if __name__ == "__main__":
    S_0 = 100; K=100; r = 0; vol = 0.2; T = 1/12
    timesteps = 30; seed = 42; n_sims = 5000
    # 몬테카를로 경로 생성하기
    paths_train = monte_carlo_paths(S_0, T, vol,r,seed,n_sims,timesteps)

    # 데이터 시각화 #
    plt.figure(figsize=(20,10))
    plt.plot(paths_train[1])
    plt.xlabel('Time Steps')
    plt.title('Stock Price Sample Paths')
    plt.show()