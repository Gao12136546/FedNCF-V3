import argparse
import json
from copy import deepcopy

import torch
import logging
import numpy as np
import pandas as pd
import random
import os
import datetime
import math
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from config import boolean_string


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


# 用于计算中间特征的均值和方差
def compute_distribution_feature(feature, eps=1e-8):
    mu = feature.mean(dim=0)
    var = feature.var(dim=0, unbiased=False)
    var = var + eps
    return mu, var



def binary_logits_kd_loss(student_logits, teacher_logits, T, eps=1e-7):
    """
    二分类推荐场景下的 soft knowledge distillation loss。

    student_prob: 学生模型输出，这里是大客户端模型输出
    teacher_prob: 教师模型输出，这里是代理模型输出
    T: 温度系数
    """

    student_logits = student_logits.view(-1)
    teacher_logits = teacher_logits.view(-1).detach()

    # 温度软化
    student_prob = torch.sigmoid(student_logits / T).clamp(eps, 1.0 - eps)
    teacher_prob = torch.sigmoid(teacher_logits / T).clamp(eps, 1.0 - eps)

    kd_loss = (
        teacher_prob * (torch.log(teacher_prob) - torch.log(student_prob))
        +
        (1.0 - teacher_prob)
        * (torch.log(1.0 - teacher_prob) - torch.log(1.0 - student_prob))
    )

    return kd_loss.mean() * (T ** 2)

def listwise_softmax_kd_loss(student_logits, teacher_logits, T):
    """
    student_logits: [B, 1] or [B]
    teacher_logits: [B, 1] or [B]
    B 表示同一个用户的一组候选 item，例如 1 个正样本 + 99 个负样本
    """

    student_logits = student_logits.view(-1)
    teacher_logits = teacher_logits.view(-1).detach()

    teacher_prob = F.softmax(teacher_logits / T, dim=0)
    student_log_prob = F.log_softmax(student_logits / T, dim=0)

    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction='sum'
    ) * (T ** 2)

    return loss

def initLogging(config):
    """Init for logging
    """
    path = config['folder_path']
    current_time = datetime.datetime.now().strftime(f'{config["title"]}_%Y-%m-%d %H-%M-%S')
    logFilename = os.path.join(path, current_time+'.txt')

    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    handlers=[
                        logging.FileHandler(logFilename, mode='w', encoding='utf-8'),
                        logging.StreamHandler()
                    ],
                    force=True
    )
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)


def load_data(config):
    # Load Data
    dataset_dir = "./data/" + config['dataset'] + "/" + "ratings.dat"
    if config['dataset'] == "ml-1m":
        rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "ml-100k":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif config['dataset'] == "baby" or config['dataset'] == "baby_10" or config['dataset'] == "baby_15" or config['dataset'] == "baby_20" or config['dataset'] == "baby_25":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "sjy-baby":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "beauty":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "sports_and_outdoors":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "toys_and_games":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "digital_music":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "cDs_and_Vinyl":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "clothing_Shoes_and_Jewelry":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "electronics":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "movies_and_TV":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    elif config['dataset'] == "tafeng":
        rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    # Reindex
    if config['dataset'] != "ali-ads":
        user_id = rating[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        rating = pd.merge(rating, user_id, on=['uid'], how='left')
        item_id = rating[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        rating = pd.merge(rating, item_id, on=['mid'], how='left')
    else:
        # ali-ads 不需要reindex
        rating['userId'] = rating['uid']
        rating['itemId'] = rating['mid']

    if config['dataset'] == 'douban' or config['dataset'] == 'bookcrossing':
        rating = rating[['userId', 'itemId', 'rating']]
    else:
        rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
    logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))

    user_id_map = user_id.set_index("userId")["uid"].to_dict()
    user_id_map = {
        value: key
        for key, value in user_id_map.items()
    }

    # 加载伪物品
    pseudo_embeddings = np.load(
        f"./data/{config['dataset']}/pseudo_item_collaborative_embeddings.npy",
        allow_pickle=False,
    ).astype(np.float32)

    pseudo_user_ids = np.load(
        f"./data/{config['dataset']}/pseudo_item_user_ids.npy",
        allow_pickle=False,
    ).astype(str)

    pseudo_item_embeddings = {
        user_id_map[element]: pseudo_embeddings[index] for index, element in enumerate(pseudo_user_ids)
    }

    return rating,pseudo_item_embeddings

# 画出结果图,全局、大客户端、小客户端三种类型。
def plot_type1(lists,type,num,config):

    if num == 10:
        index = 0
    elif num == 20:
        index = 1

    test_recalls,small_proxy_test_recalls,big_test_recalls, proxy_test_recalls, small_test_recalls = lists
    plt.figure(figsize=(10, 5))
    test_recalls = [list[index] for list in test_recalls]
    small_proxy_test_recalls = [list[index] for list in small_proxy_test_recalls]
    big_test_recalls = [list[index] for list in big_test_recalls]
    proxy_test_recalls = [list[index] for list in proxy_test_recalls]
    small_test_recalls = [list[index] for list in small_test_recalls]
    plt.plot(range(1, len(test_recalls) + 1), test_recalls,label="Small and Big", color="blue", linestyle="solid", linewidth=1)
    plt.plot(range(1, len(small_proxy_test_recalls) + 1), small_proxy_test_recalls,label="Small and Proxy", color="#DABC60", linestyle="solid", linewidth=1)
    plt.plot(range(1, len(big_test_recalls) + 1), big_test_recalls,label="Big Clients", color="red", linestyle="dashed", linewidth=1)
    plt.plot(range(1, len(proxy_test_recalls) + 1), proxy_test_recalls,label="Proxy Model", color="purple", linestyle="dashed", linewidth=1)
    plt.plot(range(1, len(small_test_recalls) + 1), small_test_recalls,label="Small Clients", color="green", linestyle="dashdot", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel(f"{type}@{num}")
    plt.title(f"ML-100K : {type}@{num}")

    step = math.floor(len(proxy_test_recalls)/20) if math.floor(len(proxy_test_recalls)/20) != 0 else 1

    plt.gca().xaxis.set_major_locator(MultipleLocator(step))
    plt.xlim(left=0)

    plt.legend()
    plt.tight_layout()

    plot_path = f"{config['folder_path']}/{config['title']}-{type}@{num}.png"
    plt.savefig(plot_path)
    plt.show()

# 计算每个客户端拥有的交互数据量
def calculate_interactions_clients(rating):
    user_interactions = rating.groupby('userId').size()
    user_interactions_dict = user_interactions.to_dict()
    return user_interactions_dict


def compute_metrics(evaluate_data, test_scores, negative_scores, Ks):
    test_users, test_items = evaluate_data[0].cpu().data.view(-1).tolist(), evaluate_data[1].cpu().data.view(-1).tolist()
    neg_users, neg_items = evaluate_data[2].cpu().data.view(-1).tolist(), evaluate_data[3].cpu().data.view(-1).tolist()
    tst_scores, neg_scores = test_scores.data.view(-1).tolist(), negative_scores.data.view(-1).tolist()
    # the golden set
    test = pd.DataFrame({'user': test_users,
                        'test_item': test_items,
                        'test_score': tst_scores})
    # the full set
    full = pd.DataFrame({'user': neg_users + test_users,
                        'item': neg_items + test_items,
                        'score': neg_scores + tst_scores})
    full = pd.merge(full, test, on=['user'], how='left')
    # rank the items according to the scores for each user
    full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
    full.sort_values(['user', 'rank'], inplace=True)
    recall, precision, ndcg = [], [], []
    for at_k in Ks:
        top_k = full[full['rank']<=at_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()  # golden items hit in the top_K items
        rec_k = len(test_in_top_k) * 1.0 / full['user'].nunique()
        # pre_k = len(test_in_top_k) * 1.0 / at_k
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        ndcg_k = test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
        recall.append(rec_k)
        ndcg.append(ndcg_k)
        
    return recall, ndcg


def to_cpu_state(state_dict):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_log_result(config, test_recalls, test_ndcgs, final_test_round):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    strr = current_time + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
        + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
        + 'best_round: ' + str(final_test_round)
    sstrr = ''
    for k in config.keys():
        sstr = ', ' + f'{k}: ' + str(config[k])
        sstrr += sstr
    strr += sstrr
    file_name = "sh_result/"+config['dataset']+".txt"
    with open(file_name, 'a') as file:
        file.write(strr + '\n')

    # logging.info('fedcs')
    logging.info('recall_list: {}'.format(test_recalls))
    # logging.info('precision_list: {}'.format(test_precisions))
    logging.info('ndcg_list: {}'.format(test_ndcgs))

    logging.info('config: {}'.format(sstrr))
    logging.info('Best test recall: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                    # test_precisions[final_test_round],
                                                                                    test_ndcgs[final_test_round],
                                                                                    final_test_round))
    logging.info('\n')


def detect_param_change(current_config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=2)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--lr_eta', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr_client', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--big_latent_dim', type=int, default=128)
    parser.add_argument('--small_latent_dim', type=int, default=64)
    parser.add_argument('--num_negative', type=int, default=4) # number of negative samples for training
    parser.add_argument('--client_model_layers', type=str, default='128,32') # 2*emb_size, emb_size, 1
    parser.add_argument('--big_client_model_layers', type=str, default='256,64') # 2*emb_size, emb_size, 1
    parser.add_argument('--small_client_model_layers', type=str, default='128,32') # 2*emb_size, emb_size, 1
    parser.add_argument('--recall_k', type=str, default='10,20')
    parser.add_argument('--l2_regularization', type=float, default=0.)
    parser.add_argument('--use_cuda', type=boolean_string, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--NUM_NEG', type=int, default=999) # number of negative samples for val&tst
    parser.add_argument('--earlystop', type=int, default=10) # number of negative samples for val&tst
    parser.add_argument('--param_agg_type', type=str, default="weighted") # 参数聚合方式
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
        config['NUM_NEG'] = 99
        config['split_point'] = 203
    elif config['dataset'] == 'baby':
        config['num_users'] = 5531
        config['num_items'] = 14328
        config['NUM_NEG'] = 99
        config['split_point'] = 13
    elif config['dataset'] == 'sjy-baby':
        config['num_users'] = 9723
        config['num_items'] = 6995
        config['NUM_NEG'] = 99
        config['split_point'] = 12
    elif config['dataset'] == 'beauty':
        config['num_users'] = 5237
        config['num_items'] = 27150
        config['NUM_NEG'] = 99
        config['split_point'] = 14
    elif config['dataset'] == 'toys_and_games':
        config['num_users'] = 5748
        config['num_items'] = 44791
        config['NUM_NEG'] = 99
        config['split_point'] = 20
    elif config['dataset'] == 'sports_and_outdoors':
        config['num_users'] = 5224
        config['num_items'] = 39877
        config['NUM_NEG'] = 99
        config['split_point'] = 20
    elif config['dataset'] == 'digital_music':
        config['num_users'] = 4575
        config['num_items'] = 31068
        config['NUM_NEG'] = 99
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

    original_config = config

    for key,value in original_config.items():
        if value != current_config[key]:
            logging.info(f"current {key} is {current_config[key]},original is {original_config[key]}")
