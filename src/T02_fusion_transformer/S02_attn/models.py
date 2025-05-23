from torch import nn
import torch
import torch.nn.functional as F


class MyModel_Attn(nn.Module):
    def __init__(
        self,
        num_tabular_features,
        ts_embedding_size,
        num_ts_features,
        lstm_num_layers,
        lstm_dropout,
        attn_n_head,
        num_output,
    ):
        super().__init__()

        self.num_tabular_features = num_tabular_features
        self.num_ts_features = num_ts_features
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.attn_n_head = attn_n_head
        self.num_output = num_output

        # Embeddeding layer (time series)
        self.ts_embedding_size = ts_embedding_size
        self.ts_embedders = nn.ModuleList(
            [nn.Linear(1, self.ts_embedding_size) for _ in range(self.num_ts_features)]
        )

        self.lstm1 = nn.LSTM(
            input_size=self.ts_embedding_size * self.num_ts_features,
            hidden_size=self.ts_embedding_size,  # This will be the output size
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.ts_embedding_size * self.num_ts_features,
            hidden_size=self.ts_embedding_size,  # This will be the output size
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
        )

        self.emb_seeder = nn.Embedding(self.num_output, self.ts_embedding_size)

        self.norm = nn.LayerNorm(self.ts_embedding_size)

        self.imh_attn = InterpretableMultiHeadAttention(
            n_head=self.attn_n_head,
            d_model=self.ts_embedding_size,
            dropout=self.lstm_dropout,
        )

        # MLP to process each output
        self.mlps_ts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ts_embedding_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                for _ in range(0, num_output)
            ]
        )

        # MLP for tabular data
        self.mlp_tab = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.regs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(64), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1)
                )
                for _ in range(0, num_output)
            ]
        )

    def forward(self, x_tab, x_ts, x_seeder):
        # x_tab: (batch, num_tabular_features)

        ts_out = torch.concat(
            [
                self.ts_embedders[i](x_ts[Ellipsis, i].unsqueeze(-1))
                for i in range(0, self.num_ts_features)
            ],
            axis=-1,
        )  # (batch, timestep, embedding_size * num_ts_features)

        ts_out, (hn, cn) = self.lstm1(ts_out)  # (batch, timestep, embedding_size)
        seeder_out = self.emb_seeder(x_seeder)  # (batch, num_output, embeding_size )
        seeder_out, (hn, cn) = self.lstm2(seeder_out)

        output = torch.concat([ts_out, seeder_out], axis=-2)

        # Layernorm
        output = self.norm(output)

        output, attn = self.imh_attn(output, output, output)

        # Extract on the the last num_output features
        output = output[:, -self.num_output :, :]

        outs = []
        for i in range(0, self.num_output):
            out_tab = self.mlp_tab(x_tab)
            out_ts = self.mlps_ts[i](output[:, i, :])
            comb = torch.cat((out_tab, out_ts), dim=1)
            out = self.regs[i](comb)
            outs.append(out)

        final = torch.concat(outs, dim=1)

        # Test with just single MLP
        # comb_size = self.ts_embedding_size * self.num_output
        # flatten = output.view(-1, comb_size)
        # outs = nn.Sequential(
        #     nn.Linear(comb_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, self.num_output),
        # )(flatten)
        # final = outs

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


# embedding_size = 12
# d_model = embedding_size
# n_head = 4
# batch_size = 12
# number_features = 6

# att_layer = InterpretableMultiHeadAttention(n_head=n_head, d_model=d_model, dropout=0.5)
# x = torch.randn(batch_size, number_features, embedding_size)
# print(x.shape)
# output, att = att_layer(x, x, x)
# print(output.shape)
# print(att.shape)
