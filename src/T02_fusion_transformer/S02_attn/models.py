from torch import nn
import torch
import torch.nn.functional as F


class NumericalEmbedder(nn.Module):
    def __init__(self, embedding_size: int, num_features: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_features = num_features
        self.embedders = nn.ModuleList(
            [nn.Linear(1, embedding_size) for _ in range(num_features)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Tensor of shape (batch, timestep, num_features)

        Returns:
            Tensor of shape (batch, timestep, embedding_size * num_features)
        """
        # Embed each feature separately and concatenate along the last axis
        embeddings = [
            self.embedders[i](X[..., i].unsqueeze(-1)) for i in range(self.num_features)
        ]
        # embeddings is a list of (batch, timestep, embedding_size)
        out = torch.cat(
            embeddings, dim=-1
        )  # (batch, timestep, embedding_size * num_features)
        return out


class MyModel_Attn(nn.Module):
    def __init__(
        self,
        tab_num_features,
        num_output,
        ts_num_features,
        ts_embedding_size=4,
        lstm_num_layers=2,
        lstm_dropout=0.5,
        attn_n_head=2,
    ):
        super().__init__()

        self.num_output = num_output
        self.ts_embedding_size = ts_embedding_size
        self.lstm_num_layers = lstm_num_layers

        # Embeddeding layer (time series)
        self.ts_embedder = NumericalEmbedder(
            embedding_size=ts_embedding_size, num_features=ts_num_features
        )

        self.lstm_encoder = nn.LSTM(
            input_size=ts_embedding_size * ts_num_features,
            hidden_size=ts_embedding_size,  # This will be the output size
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=ts_embedding_size * 1,
            hidden_size=ts_embedding_size,  # This will be the output size
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.future_embedder = NumericalEmbedder(
            num_features=1, embedding_size=ts_embedding_size
        )

        self.norm = nn.LayerNorm(ts_embedding_size)

        self.imh_attn = InterpretableMultiHeadAttention(
            n_head=attn_n_head,
            d_model=ts_embedding_size,
            dropout=lstm_dropout,
        )

        self.tab_embedder = NumericalEmbedder(
            num_features=tab_num_features, embedding_size=ts_embedding_size
        )

        # MLP for tabular data
        self.mlp_tab = nn.Sequential(
            nn.Linear(tab_num_features * ts_embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, lstm_num_layers * ts_embedding_size),
        )

        comb_size = ts_embedding_size * num_output
        self.mlp_last = nn.Sequential(
            nn.Linear(comb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_output),
        )

    def forward(self, x_tab, x_ts, x_future):
        batch_size = x_tab.shape[0]
        # x_tab: (batch, num_tabular_features)

        tab_out = self.tab_embedder(x_tab.unsqueeze(dim=1))
        tab_out = self.tab_embedder(tab_out)
        tab_out = self.mlp_tab(tab_out)
        tab_out = tab_out.view(self.lstm_num_layers, batch_size, -1)

        # (batch, timestep, embedding_size * num_ts_features)
        past_out = self.ts_embedder(x_ts)

        # (batch, timestep, embedding_size)
        past_out, (hn, cn) = self.lstm_encoder(past_out, (tab_out, tab_out))

        # (batch, num_output, embedding_size * num_ts_features)
        future_out = self.future_embedder(x_future)

        # (batch, timestep, embedding_size)
        future_out, (hn, cn) = self.lstm_decoder(future_out, (hn, cn))

        output = torch.concat([past_out, future_out], axis=-2)

        # Layer Norm
        output = self.norm(output)

        output, attn = self.imh_attn(output, output, output)

        # Extract on the the last num_output features
        output = output[:, -self.num_output :, :]

        flatten_size = batch_size * self.ts_embedding_size
        output = output.view(
            (batch_size, flatten_size)
        )  # (batch_size, num_output * embedding_size)

        flatten = output.view(batch_size, -1)
        final = self.mlp_last(flatten)
        return final, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # print('---Inputs----')
        # print('q: {}'.format(q[0]))
        # print('k: {}'.format(k[0]))
        # print('v: {}'.format(v[0]))

        attn = torch.bmm(q, k.permute(0, 2, 1))
        # print('first bmm')
        # print(attn.shape)
        # print('attn: {}'.format(attn[0]))

        if self.scale:
            dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimention
        #    print('attn_scaled: {}'.format(attn[0]))

        if mask is not None:
            # fill = torch.tensor(-1e9).to(DEVICE)
            # zero = torch.tensor(0).to(DEVICE)
            attn = attn.masked_fill(mask == 0, -1e9)
        #    print('attn_masked: {}'.format(attn[0]))

        attn = self.softmax(attn)
        # print('attn_softmax: {}'.format(attn[0]))
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)]
        )
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            # print('qs layer: {}'.format(qs.shape))
            head, attn = self.attention(qs, ks, vs, mask)
            # print('head layer: {}'.format(head.shape))
            # print('attn layer: {}'.format(attn.shape))
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        # print('concat heads: {}'.format(head.shape))
        # print('heads {}: {}'.format(0, head[0,0,Ellipsis]))
        attn = torch.stack(attns, dim=2)
        # print('concat attn: {}'.format(attn.shape))

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # print('outputs mean: {}'.format(outputs.shape))
        # print('outputs mean {}: {}'.format(0, outputs[0,0,Ellipsis]))
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


ts_embedding_size = 8
ts_num_features = 3
tab_num_features = 4
d_model = ts_embedding_size
n_head = 4
batch_size = 3
timesteps = 50
num_output = 3

model = MyModel_Attn(
    tab_num_features=tab_num_features,
    num_output=num_output,
    ts_num_features=ts_num_features,
    ts_embedding_size=ts_embedding_size,
)

x_tab = torch.randn(batch_size, tab_num_features)
x_ts = torch.randn(batch_size, timesteps, ts_num_features)
x_future = torch.randn(batch_size, num_output, 1)
(y, att) = model(x_tab=x_tab, x_future=x_future, x_ts=x_ts)
print(y.shape)
print(att.shape)


# embedder = NumericalEmbedder(
#     num_features=ts_num_features, embedding_size=ts_embedding_size
# )
# x = torch.randn(batch_size, timesteps, ts_num_features)
# y = embedder(x)
# print(x.shape)
# print(y.shape)
# print(x)
# print(y)

# att_layer = InterpretableMultiHeadAttention(n_head=n_head, d_model=d_model, dropout=0.5)
# x = torch.randn(batch_size, ts_num_features, ts_embedding_size)
# print(x.shape)
# output, att = att_layer(x, x, x)
# print(output.shape)
# print(att.shape)
