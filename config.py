import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--lr_eta', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr_client', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--big_latent_dim', type=int, default=64)
    parser.add_argument('--small_latent_dim', type=int, default=8)
    parser.add_argument('--num_negative', type=int, default=4) # number of negative samples for training
    parser.add_argument('--client_model_layers', type=str, default='16, 8') # 2*emb_size, emb_size, 1
    parser.add_argument('--big_client_model_layers', type=str, default='128, 64, 32, 16') # 2*emb_size, emb_size, 1
    parser.add_argument('--small_client_model_layers', type=str, default='16, 8') # 2*emb_size, emb_size, 1
    parser.add_argument('--distribution_adapter_hidden_dim', type=int, default=24) # 2*emb_size, emb_size, 1
    parser.add_argument('--recall_k', type=str, default='10,20')
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--big_to_proxy_weight', type=float, default=0.01)
    parser.add_argument('--distribution_kd_weight', type=float, default=0.001)
    parser.add_argument('--proxy_to_big_weight', type=float, default=1)
    parser.add_argument('--proxy_to_big_temp', type=float, default=3)
    parser.add_argument('--big_to_proxy_temp', type=float, default=1)
    parser.add_argument('--use_cuda', type=boolean_string, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=384)
    parser.add_argument('--NUM_NEG', type=int, default=999) # number of negative samples for val&tst
    parser.add_argument('--earlystop', type=int, default=100) # number of negative samples for val&tst
    parser.add_argument('--param_agg_type', type=str, default="weighted") # 参数聚合方式
    parser.add_argument('--pseudo_phrase', type=boolean_string, default=True) # 参数聚合方式
    parser.add_argument('--pseudo_loss_weight', type=float, default=1) # 参数聚合方式
    args = parser.parse_args()

    # Model.
    config = vars(args)
    if len(config['recall_k']) > 1:
        config['recall_k'] = [int(item) for item in config['recall_k'].split(',')]
    else:
        config['recall_k'] = [int(config['recall_k'])]

    if len(config['client_model_layers']) > 1:
        config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')]
        config['small_client_model_layers'] = [int(item) for item in config['small_client_model_layers'].split(',')]
        config['big_client_model_layers'] = [int(item) for item in config['big_client_model_layers'].split(',')]
    else:
        config['client_model_layers'] = int(config['client_model_layers'])
        config['small_client_model_layers'] = int(config['small_client_model_layers'])
        config['big_client_model_layers'] = int(config['big_client_model_layers'])

    if config['dataset'] == 'ml-1m':
        config['num_users'] = 6040
        config['num_items'] = 3706
        config['NUM_NEG'] = 99
        config['split_point'] = 315
    elif config['dataset'] == 'ml-100k':
        config['num_users'] = 943
        config['num_items'] = 1682
        config['NUM_NEG'] = 499
        config['split_point'] = 203
    elif config['dataset'] == 'baby':
        config['num_users'] = 5531
        config['num_items'] = 14328
        config['NUM_NEG'] = 999
        config['split_point'] = 13
    elif config['dataset'] == 'sjy-baby':
        config['num_users'] = 9723
        config['num_items'] = 6995
        config['NUM_NEG'] = 99
        config['split_point'] = 12
    elif config['dataset'] == 'beauty':
        config['num_users'] = 5237
        config['num_items'] = 27150
        config['NUM_NEG'] = 999
        config['split_point'] = 14
    elif config['dataset'] == 'toys_and_games':
        config['num_users'] = 5748
        config['num_items'] = 44791
        config['NUM_NEG'] = 999
        config['split_point'] = 20
    elif config['dataset'] == 'sports_and_outdoors':
        config['num_users'] = 5224
        config['num_items'] = 39877
        config['NUM_NEG'] = 99
        config['split_point'] = 20
    elif config['dataset'] == 'digital_music':
        config['num_users'] = 4575
        config['num_items'] = 31068
        config['NUM_NEG'] = 999
        config['split_point'] = 17
    elif config['dataset'] == 'movies_and_TV':
        config['num_users'] = 5565
        config['num_items'] = 32175
        config['NUM_NEG'] = 99
        config['split_point'] = 21
    elif config['dataset'] == 'electronics':
        config['num_users'] = 5081
        config['num_items'] = 26942
        config['NUM_NEG'] = 99
        config['split_point'] = 14
    elif config['dataset'] == 'clothing_Shoes_and_Jewelry':
        config['num_users'] = 4649
        config['num_items'] = 33731
        config['NUM_NEG'] = 99
        config['split_point'] = 13
    elif config['dataset'] == 'cDs_and_Vinyl':
        config['num_users'] = 3762
        config['num_items'] = 39892
        config['NUM_NEG'] = 99
        config['split_point'] = 24
    elif config['dataset'] == 'tafeng':
        config['num_users'] = 2486
        config['num_items'] = 16183
        config['NUM_NEG'] = 999
        config['split_point'] = 77

    return config

if __name__ == '__main__':
    config = get_config()
    print(f'./saved_model/{config["save_name"]}')