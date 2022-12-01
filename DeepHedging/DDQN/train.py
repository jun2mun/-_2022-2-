from DDQN.version2.model import DDqnModel

# ====== 파라미터 초기화 ====== #
# STOCK_CODE = "000720.KS"
# start_date = "2018-01-01"
# end_date = "2018-06-30"

STOCK_CODE = "000720.KS"
start_date = "2018-01-01"
end_date = "2019-01-01"

num_episodes = 105
batch_size = 32
update_interval = 10
save_interval = 100
plt_interval = 60

model = DDqnModel()

model.setTrainSet(STOCK_CODE, start_date, end_date)
model.train(num_episodes, batch_size, update_interval, save_interval, plt_interval,
            is_load=True, load_file_name='./save/000720.KS_500-ddqn.h5', save_file_name=STOCK_CODE)