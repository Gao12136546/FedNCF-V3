import torch.optim
from tqdm import tqdm
from fedncf import BigClient, Client
from utils import *
import numpy as np
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer(object):
    """Meta Trainer for training & evaluating NCF model

    Note: Subclass should implement self.client_model and self.server_model!
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.server_model_param = {}
        self.server_big_model_param = {} #server端保存大客户端聚合后的结果
        self.client_model_params = {}  # client 保存在local的参数
        self.proxy_model_params = {}  # 保存proxy参数
        self.big_client_optimizers = {}  # 保存大客户端的参数
        self.big_client_list = []  # 保存大客户端列表，用于参数聚合
        self.tmp_server_big_model_param = {} # 临时存储大客户端聚合参数
        self.tmp_server_proxy_model_param = {} # 临时存储代理客户端聚合参数
        self.client_crit = torch.nn.BCELoss()
        self.dis_crit = torch.nn.MSELoss()

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def is_optimizer_bound_to_model(self,optimizer, optimizer_u, optimizer_i, model):
        optimizer_param_ids = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer_param_ids.add(id(p))

        for group in optimizer_u.param_groups:
            for p in group['params']:
                optimizer_param_ids.add(id(p))

        for group in optimizer_i.param_groups:
            for p in group['params']:
                optimizer_param_ids.add(id(p))

        model_param_ids = set(id(p) for p in model.parameters())

        return optimizer_param_ids == model_param_ids

    def fed_train_pseudo_batch(
            self,
            model_client,
            pseudo_data,
            local_optimizers,
    ):
        model_client.train()
        pseudo_embeddings = pseudo_data["pseudo_embeddings"]
        if self.config['use_cuda'] is True:
            pseudo_embeddings = pseudo_embeddings.cuda()

        # 训练本地模型
        optimizer, optimizer_u, optimizer_i = local_optimizers

        optimizer.zero_grad()
        optimizer_u.zero_grad()
        pseudo_ratings = []
        for index,item_neg in enumerate(pseudo_data["neg_item_index"]):
            index = 4 - index
            pseudo_embeddings = torch.cat(
                [
                    pseudo_embeddings[:-index],  # 原来的第一行
                    model_client.embedding_item(torch.tensor(item_neg)),  # 要插入的4行
                    pseudo_embeddings[-index:],  # 原来的剩余4行
                ],
                dim=0,
            )
            pseudo_ratings.append(1)
            pseudo_ratings.extend([0] * 4)

        pseudo_ratings = torch.tensor(pseudo_ratings).float()
        pseudo_predictions = model_client(
            item_indices=[0] * pseudo_embeddings.shape[0],
            external_item_embeddings=pseudo_embeddings,
        ).view(-1)

        pseudo_loss = self.client_crit(
            pseudo_predictions,
            pseudo_ratings,
        )

        loss = self.config["pseudo_loss_weight"] * pseudo_loss
        loss.backward()
        optimizer.step()
        optimizer_u.step()

    def fed_train_single_batch(self,model_client, batch_data,local_optimizers,proxy_model = None):
        """train a batch and return an updated model."""
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        model_client.train()


        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        # 训练本地模型
        optimizer, optimizer_u, optimizer_i = local_optimizers

        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()

        ratings_pred, distribution_feature, logits = model_client(items, True, True)
        ratings_pred = ratings_pred.view(-1)
        local_loss = self.client_crit(ratings_pred, ratings)
        loss = local_loss

        if proxy_model is not None:
            # 计算大客户端的均值和方差
            big_mu, big_var = compute_distribution_feature(distribution_feature)

            # 代理模型作为 teacher，不更新参数
            proxy_model.eval()

            with torch.no_grad():
                proxy_ratings_pred, proxy_distribution_feature, proxy_logits = proxy_model(items, True, True)
                proxy_mu, proxy_var = compute_distribution_feature(proxy_distribution_feature)

            # 分布特征损失
            mean_distillation_loss = F.mse_loss(proxy_mu, big_mu)
            var_distillation_loss = F.mse_loss(torch.log(proxy_var), torch.log(big_var))
            distribution_distillation_loss = mean_distillation_loss + var_distillation_loss
            distribution_distillation_loss = distribution_distillation_loss * self.config["distribution_kd_weight"]

            # 代理模型到大客户端模型的 KD loss
            # proxy_to_big_kd_loss = binary_logits_kd_loss(
            #     student_logits= logits,
            #     teacher_logits= proxy_logits,
            #     T=self.config["proxy_to_big_temp"]
            # )
            proxy_to_big_kd_loss = listwise_softmax_kd_loss(
                student_logits= logits,
                teacher_logits= proxy_logits,
                T=self.config["proxy_to_big_temp"]
            )

            # 大客户端模型总损失
            proxy_to_big_kd_loss = self.config["proxy_to_big_weight"] * proxy_to_big_kd_loss
            loss = local_loss + proxy_to_big_kd_loss + distribution_distillation_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_client.parameters(), 5)
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()

        if proxy_model is not None:
            return model_client, proxy_model, loss.item(), proxy_to_big_kd_loss.item(), distribution_distillation_loss.item()

        return model_client, loss.item()


    def fed_train_proxy_single_batch(self, model_client, proxy_model, batch_data, proxy_optimizers):
        """train a batch and return an updated model."""
        # 初始化训练数据
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()

        proxy_model.train()
        model_client.eval()

        # 加载优化器
        optimizer, optimizer_u, optimizer_i = proxy_optimizers
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()

        # 获得大客户端的预测
        with torch.no_grad():
            big_ratings_pred,big_logits  = model_client(items, False, True)

        # 获得代理模型的预测及其均值和方差
        proxy_ratings_pred, proxy_logits = proxy_model(items, False, True)

        # 预测损失
        proxy_loss = self.client_crit(proxy_ratings_pred.view(-1), ratings)
        # logits蒸馏损失
        # logits_distillation_loss = binary_logits_kd_loss(
        #     student_logits=proxy_logits,
        #     teacher_logits=big_logits,
        #     T=self.config["big_to_proxy_temp"]
        # )

        logits_distillation_loss = listwise_softmax_kd_loss(
                student_logits=proxy_logits,
                teacher_logits=big_logits,
                T=self.config["big_to_proxy_temp"]
        )

        logits_distillation_loss = logits_distillation_loss * self.config["big_to_proxy_weight"]

        # 梯度反向传播
        loss = proxy_loss + logits_distillation_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proxy_model.parameters(), 5)
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()

        return proxy_model, loss.item(), logits_distillation_loss.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        for t, user in enumerate(round_user_params.keys()):
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data

        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

    def aggregate_big_clients_params_peruser(self, client_param, num_part, t, user):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        if user in self.big_client_list:
            if self.tmp_server_big_model_param == {}:
                for key in self.big_server_keys:
                    user_params = copy.deepcopy(client_param[key].data).cpu()
                    self.tmp_server_big_model_param[key] = user_params * self.big_clients_weight[user]
            else:
                for key in self.big_server_keys:
                    user_params = copy.deepcopy(client_param[key].data).cpu()
                    self.tmp_server_big_model_param[key].data += user_params * self.big_clients_weight[user]

        if t == num_part - 1:
            self.server_big_model_param = {}
            # 最后一个client聚合后，再把聚合好的结果保留为server_param
            for key in self.big_server_keys:
                self.server_big_model_param[key] = self.tmp_server_big_model_param[key].data
            self.tmp_server_big_model_param = {}




    def aggregate_clients_params_peruser(self, client_param, num_part, t, user):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        if user in self.proxy_model_params:
            client_param = self.proxy_model_params[user]

        if t == 0:
            self.tmp_server_model_param = {}
            # self.server_model_param = copy.deepcopy(client_param)
            for key in self.server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key] = user_params
        else:
            # print(self.server_model_param)
            for key in self.server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key].data += user_params


        if t == num_part - 1:
            self.server_model_param = {}
            # 最后一个client聚合后，再把聚合好的结果保留为server_param
            for key in self.server_keys:
                self.server_model_param[key] = self.tmp_server_model_param[key].data / num_part

    # 加权聚合
    def aggregate_clients_params_weighted(self, client_param, num_part, t, user):
        if t == 0:
            self.tmp_server_model_param = {}
            # self.server_model_param = copy.deepcopy(client_param)
            for key in self.small_server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key] = user_params * self.clients_weight[user]
        else:
            # print(self.server_model_param)
            for key in self.small_server_keys:
                user_params = copy.deepcopy(client_param[key].data).cpu()
                self.tmp_server_model_param[key].data += user_params * self.clients_weight[user]

        if t == num_part - 1:
            self.server_model_param = {}
            # 最后一个client聚合后，再把聚合好的结果保留为server_param
            for key in self.small_server_keys:
                self.server_model_param[key] = self.tmp_server_model_param[key].data

    def aggregate_proxy_clients_params_peruser(self, client_param, num_part, t, user):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        if user in self.big_client_list:
            if self.tmp_server_proxy_model_param == {}:
                for key in self.server_keys:
                    user_params = copy.deepcopy(client_param[key].data).cpu()
                    self.tmp_server_proxy_model_param[key] = user_params
            else:
                for key in self.server_keys:
                    user_params = copy.deepcopy(client_param[key].data).cpu()
                    self.tmp_server_proxy_model_param[key].data += user_params
                    if torch.isnan(self.tmp_server_model_param[key]).any():
                        print("存在 NaN")

        if t == num_part - 1:
            self.server_proxy_model_param = {}
            # 最后一个client聚合后，再把聚合好的结果保留为server_param
            for key in self.server_keys:
                self.server_proxy_model_param[key] = self.tmp_server_proxy_model_param[key].data / len(self.big_client_list)
            self.tmp_server_proxy_model_param = {}

    def get_optimizer(self,model,need_adapter):
        iemb_lr = self.config['lr_client'] * self.config['num_items'] * self.config['lr_eta'] - self.config['lr_client']
        uemb_lr = self.config['lr_client'] / self.config['clients_sample_ratio'] * self.config['lr_eta'] - self.config[
            'lr_client']

        if need_adapter:
            optimizer = torch.optim.SGD(
                [{"params": model.fc_layers.parameters()},
                 {"params": model.affine_output.parameters()},
                 {"params": model.distribution_adapter.parameters()}],
                lr=self.config['lr_client'])  # MLP optimizer
        else:
            optimizer = torch.optim.SGD(
                [{"params": model.fc_layers.parameters()},
                 {"params": model.affine_output.parameters()}],
                lr=self.config['lr_client'])  # MLP optimizer

        optimizer_u = torch.optim.SGD(model.embedding_user.parameters(),
                                      lr=uemb_lr)  # User optimizer
        # optimizer_i is responsible for updating item embedding.
        optimizer_i = torch.optim.SGD(model.embedding_item.parameters(),
                                      lr=iemb_lr)  # Item optimizer

        optimizers = [optimizer, optimizer_u, optimizer_i]

        return optimizers


    def fed_train_a_round(self, all_train_data, round_id):
        # 随机选择客户端
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(self.config['num_users'], num_participants,replace=False)  # from 0 to num_users-1
        else:
            participants = np.random.choice(self.config['num_users'], self.config['clients_sample_num'], replace=False)

        all_loss = 0
        proxy_loss = 0
        big_loss = 0
        big_to_proxy_dis_loss = 0
        proxy_to_big_dis_loss = 0
        distribution_distillation_loss = 0
        for uidx, user in tqdm(enumerate(participants), total=len(participants), desc=f"Round {round_id}"):

            model_client = copy.deepcopy(self.getClientModel(user))
            if user in self.big_client_list:
                proxy_model = copy.deepcopy(self.proxy_model)
            else:
                proxy_model = None

            # 加载参数
            if round_id != 0:
                user_param_dict = self.load_local_param(user=user)
                model_client.load_state_dict(user_param_dict)
                # 如果是大客户端的话，还需要加载大客户端对应的代理模型的参数。
                if user in self.big_client_list:
                    proxy_user_param_dict = self.proxy_load_local_param(user, proxy_model)
                    proxy_model.load_state_dict(proxy_user_param_dict)


            # 加载优化器
            if user not in self.big_client_list:
                local_optimizers = self.get_optimizer(model_client,False) # 小客户端
            else:
                local_optimizers = self.get_optimizer(model_client,True) # 大客户端
                proxy_optimizers = self.get_optimizer(proxy_model,False) # 代理模型


            # 模型训练
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            if user in self.big_client_list:
                proxy_model.train()

            # 初始化Epoch的loss
            epoch_loss = 0
            epoch_proxy_loss = 0
            epoch_big_loss = 0
            epoch_big_to_proxy_dis_loss = 0
            epoch_proxy_to_big_dis_loss = 0
            epoch_distribution_distillation_loss = 0
            # 初始化计数器
            sample_num = 0

            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    # 大客户端：本地监督训练 + 代理模型反向蒸馏
                    if user in self.big_client_list:
                        model_client, proxy_model, loss_batch, proxy_to_big_dis_loss_batch,distribution_distillation_loss_batch  = self.fed_train_single_batch(
                            model_client,
                            batch,
                            local_optimizers,
                            proxy_model=proxy_model
                        )
                    else:
                        # 小客户端：只进行普通本地训练
                        model_client, loss_batch = self.fed_train_single_batch(
                            model_client,
                            batch,
                            local_optimizers
                        )

                    # 当前Epoch的损失
                    epoch_loss += loss_batch * len(batch[0])
                    sample_num += len(batch[0])
                    # 代理模型的训练
                    if user in self.big_client_list:
                        proxy_model, proxy_loss_batch, big_to_proxy_dis_loss_batch = self.fed_train_proxy_single_batch(model_client, proxy_model, batch, proxy_optimizers)
                        epoch_big_loss += loss_batch * len(batch[0])
                        epoch_proxy_loss += proxy_loss_batch * len(batch[0])
                        epoch_big_to_proxy_dis_loss += big_to_proxy_dis_loss_batch * len(batch[0])
                        epoch_proxy_to_big_dis_loss += proxy_to_big_dis_loss_batch * len(batch[0])
                        epoch_distribution_distillation_loss += distribution_distillation_loss_batch * len(batch[0])

                if user not in self.big_client_list and self.config['pseudo_phrase']:
                    pseudo_embeddings = self.pseudo_item_embeddings.get(user)
                    pseudo_data = {
                        "pseudo_embeddings":torch.from_numpy(
                                                pseudo_embeddings
                                            ).float(),
                        "neg_item_index": self.pseudo_train_data[user]


                    }
                    if pseudo_data is not None:
                        self.fed_train_pseudo_batch(
                            model_client,
                            pseudo_data,
                            local_optimizers,
                        )

            # 计算损失
            loss = epoch_loss / sample_num
            all_loss += loss
            if user in self.big_client_list:
                user_big_loss = epoch_big_loss / sample_num
                user_proxy_loss = epoch_proxy_loss / sample_num
                user_big_to_proxy_dis_loss = epoch_big_to_proxy_dis_loss / sample_num
                user_proxy_to_big_dis_loss = epoch_proxy_to_big_dis_loss / sample_num
                user_distribution_distillation_loss = epoch_distribution_distillation_loss / sample_num

                big_loss += user_big_loss
                proxy_loss += user_proxy_loss
                big_to_proxy_dis_loss += user_big_to_proxy_dis_loss
                proxy_to_big_dis_loss += user_proxy_to_big_dis_loss
                distribution_distillation_loss += user_distribution_distillation_loss


            # 存储模型的个性化参数
            client_param = model_client.state_dict()
            self.client_model_params[user] = {}
            for key in self.client_keys:
                self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()

            # 存储代理模型的个性化参数和大客户端的个性化适配器参数
            if user in self.big_client_list:
                proxy_param = proxy_model.state_dict()
                self.proxy_model_params[user] = {}
                for key in self.client_keys:
                    self.proxy_model_params[user][key] = copy.deepcopy(proxy_param[key].data).cpu()
                for key in self.adapter_keys:
                    self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()


            # 大客户端之间聚合
            self.aggregate_big_clients_params_peruser(client_param, len(participants), uidx, user)
            # 小客户端和代理模型之间聚合
            if user in self.big_client_list:
                self.aggregate_clients_params_weighted(proxy_param, len(participants), uidx, user)
            else:
                self.aggregate_clients_params_weighted(client_param, len(participants), uidx, user)


        return all_loss / len(participants),big_loss / len(self.big_client_list), proxy_loss / len(self.big_client_list), big_to_proxy_dis_loss / len(self.big_client_list), proxy_to_big_dis_loss / len(self.big_client_list), distribution_distillation_loss / len(self.big_client_list)

    def param_client_client(self, user, client_param, save=True):
        # parameter transfer within client model
        key = 'embedding_user.weight'
        # save=True: save param from client_param -> self.client_model_params
        if save:
            self.client_model_params[user] = {}
            self.client_model_params[user][key] = copy.deepcopy(client_param[key].data).cpu()
        # save=False: load param from self.client_model_params -> client_param
        else:
            # load client local param
            if user in self.client_model_params.keys():
                client_param[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()

    def param_client_server(self, user, client_param, save=True, round_participant_params=None):
        # parameters transfer between client and server
        # save=True: save param from client_param -> round_participant_params
        nkey = 'embedding_user.weight'
        keys = [key for key in client_param.keys() if key != nkey]
        if save:
            round_participant_params[user] = {}
            for key in keys:
                round_participant_params[user][key] = copy.deepcopy(client_param[key]).data.cpu()
        # save=False: load param from self.server_model_param -> client_param
        else:
            for key in keys:
                client_param[key] = copy.deepcopy(self.server_model_param[key].data).cuda()

    def load_local_param(self, user):
        # ukey = 'embedding_user.weight'
        user_param_dict = copy.deepcopy(self.getClientModel(user).state_dict())

        if user in self.big_client_list:
            server_keys = self.big_server_keys
            for key in self.adapter_keys:
                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
        else:
            server_keys = self.small_server_keys

        for key in server_keys:
            if user not in self.big_client_list:
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data)
            else:
                user_param_dict[key] = copy.deepcopy(self.server_big_model_param[key].data)

        for key in self.client_keys:
            user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)

        return user_param_dict

    def proxy_load_local_param(self, user, proxy_model):
        # ukey = 'embedding_user.weight'
        user_param_dict = copy.deepcopy(proxy_model.state_dict())

        for key in self.small_server_keys:
            user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data)

        for key in self.client_keys:
            user_param_dict[key] = copy.deepcopy(self.proxy_model_params[user][key].data)

        return user_param_dict

    def fed_evaluate(self, evaluate_data):
        """evaluate all client models' performance using testing data."""
        y = torch.FloatTensor([1] + [0] * self.config['NUM_NEG'])
        _, test_items = evaluate_data[0], evaluate_data[1]
        _, negative_items = evaluate_data[2], evaluate_data[3]
        if self.config['use_cuda'] is True:
            test_items = test_items.cuda()
            negative_items = negative_items.cuda()
            y = y.cuda()

        test_scores, negative_scores = [], []
        proxy_test_scores, proxy_negative_scores = [], []
        per_user_losses = []
        proxy_per_user_losses = []
        for user in range(self.config['num_users']):
            if user in self.big_client_list:
                proxy_user_model = copy.deepcopy(self.proxy_model)
                user_param_dict = self.proxy_load_local_param(user, proxy_user_model)
                proxy_user_model.load_state_dict(user_param_dict)
                proxy_user_model.eval()

                user_model = copy.deepcopy(self.getClientModel(user))
                user_param_dict = self.load_local_param(user)
            else:
                user_model = copy.deepcopy(self.getClientModel(user))
                user_param_dict = self.load_local_param(user)

            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            with torch.no_grad():
                # obtain user's positive test information.
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_item = negative_items[user * self.config['NUM_NEG']: (user + 1) * self.config['NUM_NEG']]
                # perform model prediction.
                if user in self.big_client_list:
                    proxy_test_score = proxy_user_model(test_item)
                    proxy_negative_score = proxy_user_model(negative_item)

                    proxy_test_scores.append(proxy_test_score)
                    proxy_negative_scores.append(proxy_negative_score)

                    test_score = user_model(test_item)
                    negative_score = user_model(negative_item)
                else:
                    test_score = user_model(test_item)
                    negative_score = user_model(negative_item)

                    proxy_test_scores.append(test_score)
                    proxy_negative_scores.append(negative_score)

                y_hat = torch.cat((test_score, negative_score))
                loss = self.client_crit(y_hat.view(-1), y)

                if user in self.big_client_list:
                    proxy_y_hat = torch.cat((proxy_test_score, proxy_negative_score))
                    proxy_loss = self.client_crit(proxy_y_hat.view(-1), y)
                    proxy_per_user_losses.append(proxy_loss.item())
                else:
                    proxy_per_user_losses.append(loss.item())

                test_scores.append(test_score)
                negative_scores.append(negative_score)
                per_user_losses.append(loss.item())

        test_scores = torch.cat(test_scores).cpu()
        negative_scores = torch.cat(negative_scores).cpu()

        proxy_test_scores = torch.cat(proxy_test_scores).cpu()
        proxy_negative_scores = torch.cat(proxy_negative_scores).cpu()

        recall, ndcg = compute_metrics(evaluate_data, test_scores, negative_scores, self.config['recall_k'])
        avg_loss = sum(per_user_losses) / self.config['num_users']

        return recall, ndcg, avg_loss, {
            'test_scores': test_scores,
            'negative_scores': negative_scores,
            'per_user_losses': per_user_losses,
            'proxy_test_scores': proxy_test_scores,
            'proxy_negative_scores': proxy_negative_scores,
            'proxy_per_user_losses': proxy_per_user_losses
        }


    def fed_evaluate_small_proxy(self, evaluate_data, evaluate_details):
        test_scores = evaluate_details['proxy_test_scores']
        negative_scores = evaluate_details['proxy_negative_scores']
        per_user_losses = evaluate_details['proxy_per_user_losses']

        # compute the evaluation metrics.
        recall, ndcg = compute_metrics(evaluate_data, test_scores, negative_scores, self.config['recall_k'])
        avg_loss = sum(per_user_losses) / self.config['num_users']
        return recall, ndcg, avg_loss

    def fed_evaluate_separate(self, evaluate_data, user_list, is_proxy, evaluate_details):
        """evaluate all client models' performance using testing data."""
        if is_proxy:
            test_scores = evaluate_details['proxy_test_scores'][user_list]
            negative_scores = torch.cat([
                evaluate_details['proxy_negative_scores'][i * self.config["NUM_NEG"]:(i + 1) * self.config["NUM_NEG"]]
                for i in user_list
            ])
            per_user_losses = [evaluate_details['proxy_per_user_losses'][i] for i in user_list]
        else:
            test_scores = evaluate_details['test_scores'][user_list]
            negative_scores = torch.cat([
                evaluate_details['negative_scores'][i * self.config["NUM_NEG"]:(i + 1) * self.config["NUM_NEG"]]
                for i in user_list
            ])
            per_user_losses = [evaluate_details['per_user_losses'][i] for i in user_list]

        evaluate_data_tmp = self._slice_evaluate_data(evaluate_data, user_list)
        recall, ndcg = compute_metrics(evaluate_data_tmp, test_scores, negative_scores, self.config['recall_k'])
        return recall, ndcg, sum(per_user_losses) / len(user_list)

    def get_params(self):
        save_params = {
            'server': copy.deepcopy(self.server_model_param),
            'server_big_model': copy.deepcopy(self.server_big_model_param),
            'proxy_model': copy.deepcopy(self.proxy_model_params),
            'client': copy.deepcopy(self.client_model_params)
        }
        return save_params

    def _slice_evaluate_data(self, evaluate_data, user_list):
        """Build evaluate_data for a subset of users without recomputing predictions."""
        user_list = list(user_list)
        evaluate_data_tmp = [tensor.clone() for tensor in evaluate_data]
        evaluate_data_tmp[0], evaluate_data_tmp[1] = evaluate_data[0][user_list], evaluate_data[1][user_list]
        evaluate_data_tmp[2] = torch.cat([
            evaluate_data[2][i * self.config["NUM_NEG"]:(i + 1) * self.config["NUM_NEG"]]
            for i in user_list
        ])
        evaluate_data_tmp[3] = torch.cat([
            evaluate_data[3][i * self.config["NUM_NEG"]:(i + 1) * self.config["NUM_NEG"]]
            for i in user_list
        ])
        return evaluate_data_tmp

    def run_experiment(self, config, sample_generator):
        test_recalls, test_ndcgs = [], []
        small_proxy_test_recalls, small_proxy_test_ndcgs = [], []
        big_test_recalls, big_test_ndcgs = [], []
        proxy_test_recalls, proxy_test_ndcgs = [], []
        small_test_recalls, small_test_ndcgs = [], []
        best_recall, final_test_round, best_param = 0, 0, None
        test_data = sample_generator.test_data

        for round in range(config['num_round']):
            if config['pseudo_phrase']:
                if round == 1:
                    checkpoint_path = f"./data/{config['dataset']}/{config['dataset']}_best_param.pt"
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                    )
                    self.server_model_param = checkpoint["server_model_param"]
                    self.server_big_model_param = checkpoint[
                        "server_big_model_param"
                    ]
                    self.client_model_params = checkpoint[
                        "client_model_params"
                    ]
                    self.proxy_model_params = checkpoint[
                        "proxy_model_params"
                    ]


            # break
            logging.info('-' * 80)
            logging.info('-' * 80)
            logging.info('Round {} starts !'.format(round))

            logging.info('-' * 80)
            logging.info('Training phase!')  # 每一个epoch都要重新sample
            all_train_data = sample_generator.store_all_train_data(config['num_negative'])
            train_loss, big_loss, proxy_loss, big_to_proxy_dis_loss, proxy_to_big_dis_loss, distribution_distillation_loss = self.fed_train_a_round(all_train_data, round)
            logging.info('Trn_Loss={:.5f}'.format(train_loss))
            logging.info('Big_Loss={:.5f}'.format(big_loss))
            logging.info('Proxy_Loss={:.5f}'.format(proxy_loss))
            logging.info('Big_to_Proxy_Dis_Loss={:.5f}'.format(big_to_proxy_dis_loss))
            logging.info('Proxy_to_Big_Dis_Loss={:.5f}'.format(proxy_to_big_dis_loss))
            logging.info('Distribution_Loss={:.5f}'.format(distribution_distillation_loss))

            logging.info('-' * 80)
            logging.info('Testing phase-Small and Big!')
            test_recall, test_ndcg, test_loss, evaluate_details = self.fed_evaluate(test_data)
            logging.info(result2str('Recall', config['recall_k'], test_recall))
            logging.info(result2str('NDCG', config['recall_k'], test_ndcg))
            logging.info('Tst_Loss={:.5f}'.format(test_loss))
            test_recalls.append(test_recall)
            test_ndcgs.append(test_ndcg)

            logging.info('-' * 80)
            logging.info('Testing phase-Small and Proxy!')
            small_proxy_test_recall, small_proxy_test_ndcg, small_proxy_test_loss = self.fed_evaluate_small_proxy(test_data, evaluate_details)
            logging.info(result2str('Recall', config['recall_k'], small_proxy_test_recall))
            logging.info(result2str('NDCG', config['recall_k'], small_proxy_test_ndcg))
            logging.info('Tst_Loss={:.5f}'.format(small_proxy_test_loss))
            small_proxy_test_recalls.append(small_proxy_test_recall)
            small_proxy_test_ndcgs.append(small_proxy_test_ndcg)


            logging.info('-' * 80)
            logging.info('Separate Testing phase!')
            big_users_list = [user_id for user_id, cnt in self.interactions_client.items() if cnt >= config['split_point']]
            small_users_list = [user_id for user_id, cnt in self.interactions_client.items() if cnt < config['split_point']]
            big_test_recall, big_test_ndcg, big_test_loss = self.fed_evaluate_separate(test_data, big_users_list, False, evaluate_details)
            proxy_test_recall, proxy_test_ndcg, proxy_test_loss = self.fed_evaluate_separate(test_data, big_users_list, True, evaluate_details)
            small_test_recall, small_test_ndcg, small_test_loss = self.fed_evaluate_separate(test_data, small_users_list, False, evaluate_details)
            logging.info(result2str('Big Clients:Recall', config['recall_k'], big_test_recall))
            logging.info(result2str('Big NDCG', config['recall_k'], big_test_ndcg))
            logging.info(result2str('Proxy Model:Recall', config['recall_k'], proxy_test_recall))
            logging.info(result2str('Proxy NDCG', config['recall_k'], proxy_test_ndcg))
            logging.info(result2str('Small Clients:Recall', config['recall_k'], small_test_recall))
            logging.info(result2str('Small NDCG', config['recall_k'], small_test_ndcg))
            big_test_recalls.append(big_test_recall)
            big_test_ndcgs.append(big_test_ndcg)
            proxy_test_recalls.append(proxy_test_recall)
            proxy_test_ndcgs.append(proxy_test_ndcg)
            small_test_recalls.append(small_test_recall)
            small_test_ndcgs.append(small_test_ndcg)

            # 按照HR@10早停
            if test_recall[0] >= best_recall:
                best_recall = test_recall[0]
                final_test_round = round
                cnt = 0

                item_checkpoint = {
                    "item_embedding_weight": (
                        self.server_model_param['embedding_item.weight'].detach().cpu().clone()
                    ),
                    "item_id_map":self.item_id_map
                }

                checkpoint = {
                    "stage": "warmup_complete",
                    "round": final_test_round,
                    "server_model_param": to_cpu_state(self.server_model_param),
                    "server_big_model_param": to_cpu_state(
                        self.server_big_model_param
                    ),
                    "client_model_params": {
                        user_id: to_cpu_state(model_state)
                        for user_id, model_state in self.client_model_params.items()
                    },
                    "proxy_model_params": {
                        user_id: to_cpu_state(model_state)
                        for user_id, model_state in self.proxy_model_params.items()
                    },
                    "config": copy.deepcopy(self.config),
                }

                torch.save(item_checkpoint, f"./log/{config['dataset']}_col_emb_item.pt")
                torch.save(checkpoint, f"./log/{config['dataset']}_best_param.pt")
                print("保存了参数")

            else:
                cnt += 1
                logging.info(f'Early stop at: {cnt} out of {config["earlystop"]}')
                if cnt >= config['earlystop']:
                    break

        plot_type1([test_recalls, small_proxy_test_recalls,big_test_recalls, proxy_test_recalls, small_test_recalls], "HR", 10, config)
        plot_type1([test_ndcgs, small_proxy_test_ndcgs,big_test_ndcgs, proxy_test_ndcgs,small_test_ndcgs], "NDCG", 10, config)
        plot_type1([test_recalls, small_proxy_test_recalls, big_test_recalls, proxy_test_recalls,small_test_recalls], "HR", 20, config)
        plot_type1([test_ndcgs, small_proxy_test_ndcgs,big_test_ndcgs, proxy_test_ndcgs,small_test_ndcgs], "NDCG", 20, config)

        return test_recalls, test_ndcgs, final_test_round


def result2str(metric, Ks, results):
    return '{}@{} = {:.6f}, {}@{} = {:.6f}'.format(
        metric, Ks[0], results[0],
        metric, Ks[1], results[1])\



class FedTrainer(Trainer):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.interactions_client = None
        self.client_model = Client(config)
        self.small_client_model = copy.deepcopy(self.client_model)
        self.proxy_model = copy.deepcopy(self.client_model)
        self.big_client_model = BigClient(config)

        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.client_model.cuda()
            self.small_client_model.cuda()
            self.proxy_model.cuda()
            self.big_client_model.cuda()

        self.small_mlp_keys = [k for k in self.client_model.state_dict().keys() if
                         k.split('.')[0] in ['fc_layers', 'affine_output']]

        self.big_mlp_keys = [k for k in self.big_client_model.state_dict().keys() if
                             k.split('.')[0] in ['fc_layers', 'affine_output']]

        self.adapter_keys = [k for k in self.big_client_model.state_dict().keys() if
                         k.split('.')[0] in ['distribution_adapter']]

        self.small_server_keys = ['embedding_item.weight'] + self.small_mlp_keys  # param saved in server
        self.big_server_keys = ['embedding_item.weight'] + self.big_mlp_keys  # param saved in server
        self.client_keys = ['embedding_user.weight'] # param saved in client
        self.adapter_keys = self.adapter_keys
        super(FedTrainer, self).__init__(config)

    def getClientModel(self, user):
        if self.interactions_client[user] >= self.config["split_point"]:
            return self.big_client_model
        else:
            return self.small_client_model
