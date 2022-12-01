from DDQN.version2.model import DDqnModel



# ====== 파라미터 초기화 ====== #
STOCK_CODE = "000720.KS"
start_date = "2018-01-01"
end_date = "2019-01-01"

model = DDqnModel()

model.setTrainSet(STOCK_CODE, start_date, end_date)
model.predict("./save/000720.KS_500-ddqn.h5")
# model.train(num_episodes, batch_size, update_interval, save_interval, plt_interval,
#             is_load=False, load_file_name=None, save_file_name=STOCK_CODE)