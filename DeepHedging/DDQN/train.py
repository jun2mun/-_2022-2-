from DDQN.model import DDqnModel



# ====== 파라미터 초기화 ====== #
STOCK_CODE = "^KS11"
start_date = "2019-01-01"
end_date = "2020-03-30"

num_episodes = 1000
batch_size = 32
update_interval = 10
save_interval = 300
plt_interval = 60

model = DDqnModel()

model.setTrainSet(STOCK_CODE, start_date, end_date)
model.train(num_episodes, batch_size, update_interval, save_interval, plt_interval,
            is_load=False, load_file_name=None, save_file_name=STOCK_CODE)