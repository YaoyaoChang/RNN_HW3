import torch.nn as nn
import torch
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout_input=0.5, dropout_rnn=0.5, dropout_decoder=0.5, tie_weights=False, attention=False):
        super(RNNModel, self).__init__()
        self.drop_input = nn.Dropout(dropout_input)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout_rnn)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout_rnn)
        self.attention = attention
        if self.attention:
            self.attention_module = Attention(nhid)
            self.concat_layer = nn.Linear(nhid*2, nhid)
        self.drop_decoder = nn.Dropout(dropout_decoder)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
#         print("debug:input.shape=",input.shape) # [35, 20]
        emb = self.drop_input(self.encoder(input))
#         print("debug:emb.shape=",emb.shape) # [35, 20, 200]
        output, hidden = self.rnn(emb, hidden)
#         print("debug:output.shape=",output.shape) # [35, 20, 200]
#         print("debug:hidden.shape=",hidden.shape)
        if self.attention:
            context_vectors, attention_score = self.attention_module(output)
#             print("debug:context_vectors.shape=",context_vectors.shape) # [35, 20, 200]
            combine_encoding = torch.cat((context_vectors, output), dim=2)
#             print("debug:combine_encoding.shape=",combine_encoding.shape) # [35, 20, 400]
            output = torch.tanh(self.concat_layer(combine_encoding))
#             print("debug:output.shape=",output.shape) # [35, 20, 200]
        output = self.drop_decoder(output)
#         print("debug:output.shape=",output.shape) # [35, 20, 200]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
#         print("debug:decoded.shape=",decoded.shape) # [700, 10000]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """
        sequence_length = x.shape[1]
#         sequence_length = x.shape[0]
        
        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))
#         print("debug:self_attention_scores.shape=",self_attention_scores.shape) # [35, 20, 1]
        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)
#             weighted_attention_scores = F.softmax(
#                 self_attention_scores[:t+1, :, :].clone(), dim=1)
#             print("debug:weighted_attention_scores.shape=",weighted_attention_scores.shape) # [35, t, 1] [t, 20, 1] (where t is 'for t in range(sequence_length) 

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t+1, :].clone(), dim=1))
#             context_vectors.append(
#                 torch.sum(weighted_attention_scores * x[:t+1, :, :].clone(), dim=0))
#             print("debug:context_vectors[len(context_vectors)-1].shape=",context_vectors[len(context_vectors)-1].shape) # [35, 200] [20, 200]

            if return_attention:
                attention_vectors.append(
                    weighted_attention_scores.cpu().detach().numpy())

#         print("debug:weighted_attention_scores.shape=",weighted_attention_scores.shape) # [35, 20, 1]

        context_vectors = torch.stack(context_vectors).transpose(0, 1)
#         context_vectors = torch.stack(context_vectors)
#         print("debug:context_vectors.shape=",context_vectors.shape) # [35, 20, 200]

        return context_vectors, attention_vectors