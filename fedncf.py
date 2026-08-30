import torch
from adapter import DistributionFeatureAdapter


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()

        config['client_model_layers'] = config[f'small_client_model_layers']
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config[f'small_latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()



    def forward(self, item_indices, is_proxy = False, return_logits = False, external_item_embeddings=None,):
        if self.config['use_cuda']:
            user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cuda())
        else:
            user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)))

        if external_item_embeddings is None:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = external_item_embeddings

        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.LeakyReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        if is_proxy and return_logits:
            return rating, vector, logits
        elif is_proxy and not return_logits:
            return rating, vector
        elif not is_proxy and return_logits:
            return rating, logits
        elif not is_proxy and not return_logits:
            return rating


    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass



class BigClient(torch.nn.Module):
    def __init__(self, config):
        super(BigClient, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['big_latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['big_client_model_layers'][:-1], config['big_client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['big_client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # 小模型最后一层特征维度
        small_feature_dim = config['small_client_model_layers'][-1]

        # 大模型最后一层特征维度
        big_feature_dim = config['big_client_model_layers'][-1]

        # 分布特征的适配器
        self.distribution_adapter = DistributionFeatureAdapter(
            original_dim= big_feature_dim,
            aim_dim= small_feature_dim,
            hidden_dim=config['distribution_adapter_hidden_dim']
        )

    def forward(self, item_indices, return_feature = False, return_logits = False):
        if self.config['use_cuda']:
            user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cuda())
        else:
            user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)))
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.LeakyReLU()(vector)

        # 中间层特征
        hidden_feature = vector

        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        if return_feature and return_logits:
            distribution_feature = self.distribution_adapter(hidden_feature)
            return rating, distribution_feature, logits
        elif return_feature and not return_logits:
            distribution_feature = self.distribution_adapter(hidden_feature)
            return rating, distribution_feature
        elif not return_feature and return_logits:
            return rating, logits
        elif not return_feature and not return_logits:
            return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass






