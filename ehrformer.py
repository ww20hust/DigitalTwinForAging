import torch.nn as nn
import torch
from einops import rearrange
from transformers import BertConfig, GPT2Model, BertModel
from transformers.models.bert.modeling_bert import BertEncoder


class MultiTaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['output_dim']
        self.proj_dim = config['proj_dim']

        self.cls_fc = nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, 2)
        )
        self.reg_fc = nn.Sequential(
            nn.Linear(self.output_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, 1)
        )

    def forward(self, h_cls, h_reg):
        # h_cls = h[:, :self.n_cls, :]
        # h_reg = h[:, self.n_cls:, :]
        y_cls = self.cls_fc(h_cls)
        y_reg = self.reg_fc(h_reg).squeeze(2)
        return y_cls, y_reg


class EHREmbedding2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer(
            "token_type_ids", torch.arange(config['n_category_feats']+config['n_float_feats']), persistent=False
        )
        self.register_buffer(
            "one", torch.ones(1, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "zero", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.pool_emb = self.config['pool_emb']
        self.bert = BertModel(self.config['BERT_config'])

    def forward(self, cat_feats, float_feats):  # cat_feats: (b, nc), float_feats: (b, nf)
        B = cat_feats.shape[0]
        L = cat_feats.shape[2]

        cat_feats_mask = cat_feats == -1
        float_feats_mask = float_feats == -1

        attention_mask = torch.cat([self.one.unsqueeze(1).unsqueeze(0).expand(B, -1, L), ~cat_feats_mask, ~float_feats_mask], dim=1)
        attention_mask = rearrange(attention_mask, 'b n l -> (b l) n')

        cat_feats = cat_feats + 2
        cat_feats[cat_feats_mask] = 1
        float_feats = float_feats + 2 + self.config['n_category_values']
        float_feats[float_feats_mask] = 1
        input_ids = torch.cat([self.zero.unsqueeze(1).unsqueeze(0).expand(B, -1, L), cat_feats, float_feats], dim=1)
        input_ids = rearrange(input_ids, 'b n l -> (b l) n')

        # attention_mask = torch.cat([~cat_feats_mask, ~float_feats_mask], dim=1)
        # cat_feats = cat_feats + 2
        # cat_feats[cat_feats_mask] = 1
        # float_feats = float_feats + 2 + self.config['n_category_values']
        # float_feats[float_feats_mask] = 1
        # input_ids = torch.cat([cat_feats, float_feats], dim=1)
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(B * L, -1)
        ft_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.pool_emb:
            ft_emb = ft_emb.pooler_output
        else:
            ft_emb = ft_emb.last_hidden_state
            ft_emb = rearrange(ft_emb, '(b l) n d -> b l n d', b=B)

        return ft_emb  # time_index: (b, d)


class EHRVAE2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_cls = len(self.config['cls_label_names'])
        self.n_reg = len(self.config['reg_label_names'])

        self.config['BERT_config'] = BertConfig(
            vocab_size=config['n_category_values']+config['n_float_values']+2,  # 1 for padding, 0 for CLS
            hidden_size=config['output_dim'],
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=config['output_dim'] * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,  # 0 for CLS
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None
        )
        self.ehr_layer = EHREmbedding2D(config)
        self.gpt = GPT2Model(config['transformer'])
        self.ehr_mu = BertEncoder(self.config['BERT_config'])
        self.ehr_std = BertEncoder(self.config['BERT_config'])
        self.decoder = BertEncoder(self.config['BERT_config'])
        self.head = MultiTaskHead(config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cat_feats, float_feats, time_index, valid_mask):
        feat = self.ehr_layer(cat_feats, float_feats)
        cls_feat = feat[:, :, 0, :]
        rest_feat = feat[:, :, 1:, :]
        y = self.gpt(
            inputs_embeds=cls_feat,
            position_ids=time_index,
            attention_mask=valid_mask
        ).last_hidden_state.unsqueeze(dim=2)
        feat = torch.cat([y, rest_feat], dim=2)
        feat = rearrange(feat, 'b l n d -> (b l) n d')
        mu_z = self.ehr_mu(feat).last_hidden_state
        std_z = self.ehr_std(feat).last_hidden_state
        z = self.reparameterize(mu_z, std_z)
        h = self.decoder(z).last_hidden_state # B, Nc+Nf, D
        h = h[:, 1:, :]
        h_cls = h[:, :self.n_cls, :]
        h_reg = h[:, self.n_cls:, :]
        y_cls, y_reg = self.head(h_cls, h_reg)
        return y_cls, y_reg, mu_z, std_z
    
    def forward_mu(self, cat_feats, float_feats):
        mu_z = self.ehr_mu(cat_feats, float_feats)
        return mu_z
    
class EHREmbedding1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer(
            "token_type_ids", torch.arange(config['n_category_feats']+config['n_float_feats']+1), persistent=False
        )
        self.register_buffer(
            "one", torch.ones(1, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "zero", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.pool_emb = self.config['pool_emb']
        self.bert = BertModel(self.config['BERT_config'])

    def forward(self, cat_feats, float_feats):  # cat_feats: (b, nc), float_feats: (b, nf)
        B = cat_feats.shape[0]

        cat_feats_mask = cat_feats == -1
        float_feats_mask = float_feats == -1

        attention_mask = torch.cat([self.one.unsqueeze(0).expand(B, -1), ~cat_feats_mask, ~float_feats_mask], dim=1)

        cat_feats = cat_feats + 2
        cat_feats[cat_feats_mask] = 1
        float_feats = float_feats + 2 + self.config['n_category_values']
        float_feats[float_feats_mask] = 1
        input_ids = torch.cat([self.zero.unsqueeze(0).expand(B, -1), cat_feats, float_feats], dim=1)

        # attention_mask = torch.cat([~cat_feats_mask, ~float_feats_mask], dim=1)
        # cat_feats = cat_feats + 2
        # cat_feats[cat_feats_mask] = 1
        # float_feats = float_feats + 2 + self.config['n_category_values']
        # float_feats[float_feats_mask] = 1
        # input_ids = torch.cat([cat_feats, float_feats], dim=1)
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(B, -1)
        ft_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.pool_emb:
            ft_emb = ft_emb.pooler_output
        else:
            ft_emb = ft_emb.last_hidden_state
        return ft_emb  # time_index: (b, d)
    
class EHRVAE1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_cls = len(self.config['cls_label_names'])
        self.n_reg = len(self.config['reg_label_names'])

        self.config['BERT_config'] = BertConfig(
            vocab_size=config['n_category_values']+config['n_float_values']+2,  # 1 for padding, 0 for CLS
            hidden_size=config['output_dim'],
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=config['output_dim'] * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,  # 0 for CLS
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None
        )
        self.ehr_layer = EHREmbedding1D(config)
        self.ehr_mu = BertEncoder(self.config['BERT_config'])
        self.ehr_std = BertEncoder(self.config['BERT_config'])
        self.decoder = BertEncoder(self.config['BERT_config'])
        self.head = MultiTaskHead(config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cat_feats, float_feats):
        feat = self.ehr_layer(cat_feats, float_feats)
        mu_z = self.ehr_mu(feat).last_hidden_state
        std_z = self.ehr_std(feat).last_hidden_state
        z = self.reparameterize(mu_z, std_z)
        h = self.decoder(z).last_hidden_state # B, Nc+Nf, D
        h = h[:, 1:, :]
        h_cls = h[:, :self.n_cls, :]
        h_reg = h[:, self.n_cls:, :]
        y_cls, y_reg = self.head(h_cls, h_reg)
        return y_cls, y_reg, mu_z, std_z
    
    def forward_mu(self, cat_feats, float_feats):
        mu_z = self.ehr_mu(cat_feats, float_feats)
        return mu_z
    
