import os
from trainer import FedTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from utils import *
from config import *
from data import SampleGenerator
from itertools import product

config = get_config()

SEARCH_SPACE = {
    "big_to_proxy_temp": [1,2,3,4],
    "proxy_to_big_temp":[1,2,3,4]
}

search_keys = list(SEARCH_SPACE.keys())
param_settings = list(product(*[SEARCH_SPACE[k] for k in search_keys]))

for setting_idx, param in enumerate(param_settings):
    trial_config = dict(config)
    for k, v in zip(search_keys, param):
        trial_config[k] = v

    if trial_config['dataset'] == 'ml-1m':
        trial_config['num_users'] = 6040
        trial_config['num_items'] = 3706
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 315
    elif trial_config['dataset'] == 'ml-100k':
        trial_config['num_users'] = 943
        trial_config['num_items'] = 1682
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 203
    elif trial_config['dataset'] == 'baby':
        trial_config['num_users'] = 5531
        trial_config['num_items'] = 14328
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 13
    elif trial_config['dataset'] == 'sjy-baby':
        trial_config['num_users'] = 9723
        trial_config['num_items'] = 6995
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 12
    elif trial_config['dataset'] == 'beauty':
        trial_config['num_users'] = 5237
        trial_config['num_items'] = 27150
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 14
    elif trial_config['dataset'] == 'toys_and_games':
        trial_config['num_users'] = 5748
        trial_config['num_items'] = 44791
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 20
    elif trial_config['dataset'] == 'sports_and_outdoors':
        trial_config['num_users'] = 5224
        trial_config['num_items'] = 39877
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 20
    elif trial_config['dataset'] == 'digital_music':
        trial_config['num_users'] = 4575
        trial_config['num_items'] = 31068
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 17
    elif trial_config['dataset'] == 'movies_and_TV':
        trial_config['num_users'] = 5565
        trial_config['num_items'] = 32175
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 21
    elif trial_config['dataset'] == 'electronics':
        trial_config['num_users'] = 5081
        trial_config['num_items'] = 26942
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 14
    elif trial_config['dataset'] == 'clothing_Shoes_and_Jewelry':
        trial_config['num_users'] = 4649
        trial_config['num_items'] = 33731
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 13
    elif trial_config['dataset'] == 'cDs_and_Vinyl':
        trial_config['num_users'] = 3762
        trial_config['num_items'] = 39892
        trial_config['NUM_NEG'] = 99
        trial_config['split_point'] = 24

    trial_config['latent_dim'] = int(trial_config['client_model_layers'][0]/2)

    title = (f"{list(SEARCH_SPACE.keys())[0]}{trial_config[list(SEARCH_SPACE.keys())[0]]}_"
             f"{list(SEARCH_SPACE.keys())[1]}{trial_config[list(SEARCH_SPACE.keys())[1]]}")

    folder_path = f"./grid_search_result/{title}"
    trial_config['folder_path'] = folder_path
    trial_config['title'] = title
    os.makedirs(folder_path, exist_ok=True)

    initLogging(trial_config)
    logging.info(trial_config)

    seed_all(trial_config['seed'])

    trainer = FedTrainer(trial_config)

    rating = load_data(trial_config)
    trainer.interactions_client = calculate_interactions_clients(rating)

    total_interactions = sum(trainer.interactions_client.values())
    trainer.clients_weight = {k: v / total_interactions for k, v in trainer.interactions_client.items()}
    trainer.big_client_list = [k for k, v in trainer.interactions_client.items() if v >= trial_config['split_point']]

    big_total_interactions = sum([v for k, v in trainer.interactions_client.items() if k in trainer.big_client_list])
    trainer.big_clients_weight = {k: v / big_total_interactions for k, v in trainer.interactions_client.items() if
                                  k in trainer.big_client_list}

    # DataLoader for training
    sample_generator = SampleGenerator(config=trial_config, ratings=rating)

    detect_param_change(trial_config)

    test_recalls, test_ndcgs, final_test_round = trainer.run_experiment(trial_config, sample_generator)

    save_log_result(trial_config, test_recalls, test_ndcgs, final_test_round)