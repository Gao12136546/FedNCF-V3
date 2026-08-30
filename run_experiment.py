import os

from trainer import FedTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
from config import *
from data import SampleGenerator

config = get_config()
print(config)
config["title"] = f"FedNCF_{config['dataset']}"
config['folder_path'] = f"./log"

seed_all(config['seed'])

trainer = FedTrainer(config)

# 打印model结构
print(trainer.client_model)

initLogging(config)
logging.info(config)

rating, pseudo_item_embeddings = load_data(config)
trainer.pseudo_item_embeddings = pseudo_item_embeddings
trainer.interactions_client = calculate_interactions_clients(rating)

total_interactions = sum(trainer.interactions_client.values())
trainer.clients_weight = { k : v / total_interactions for k,v in trainer.interactions_client.items()}
trainer.big_client_list = [k for k,v in trainer.interactions_client.items() if v >= config['split_point']]

big_total_interactions = sum([v for k,v in trainer.interactions_client.items() if k in trainer.big_client_list])
trainer.big_clients_weight = { k : v / big_total_interactions for k,v in trainer.interactions_client.items() if k in trainer.big_client_list}

sample_generator = SampleGenerator(config=config, ratings=rating)
sample_generator.pseudo_item_embeddings = pseudo_item_embeddings
trainer.pseudo_train_data = sample_generator.store_pseudo_train_data(4)
detect_param_change(config)

test_recalls, test_ndcgs, final_test_round = trainer.run_experiment(config, sample_generator)

save_log_result(config, test_recalls, test_ndcgs, final_test_round)