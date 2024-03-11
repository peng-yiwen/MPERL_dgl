import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


class PonderRelationalGraphConvModel(nn.Module):
    def __init__(self, embedding_dim, num_nodes, num_rels, num_types, self_loop=False, 
                 regularizer='basis', num_bases=-1, num_layers=2, activation='relu', 
                 bias=False, dropout=0.2, max_steps=1, seed=0, cuda=False):
        
        torch.manual_seed(seed)
        super(PonderRelationalGraphConvModel, self).__init__()

        self.num_rels = num_rels
        self.num_nodes = num_nodes

        self.max_steps = max_steps
        self.dropout = dropout
        self.is_halt = False
        self.lambda_layer = nn.Linear(num_types, 1)
        self.lambda_prob = nn.Sigmoid()
        # self.softmax = nn.Softmax()
        # An option to set during inference so that computation is actually halted at inference time

        self.conv_layers = nn.ModuleList()
        if activation=='none':
            activation_fn = None
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('%s is not supported' % activation)
        

        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(
                    RGCNLayer(embedding_dim, embedding_dim, 2*num_rels, self_loop=self_loop,
                              regularizer=regularizer, num_bases=num_bases, activation=activation_fn, dropout=dropout, bias=bias)
                )
            else:
                if i == num_layers-1: # the last layer
                    self.conv_layers.append(
                        RGCNLayer(embedding_dim, num_types, 2*num_rels, self_loop=self_loop,
                                  regularizer=regularizer, num_bases=num_bases, activation=None, dropout=dropout, bias=bias)
                    )
                else:
                    self.conv_layers.append(
                        RGCNLayer(embedding_dim, embedding_dim, 2*num_rels, self_loop=self_loop,
                                  regularizer=regularizer, num_bases=num_bases, activation=activation_fn, dropout=dropout, bias=bias)
                    )
        
        # # Option1: initilization using uniform distribution
        # self.entity = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        # nn.init.uniform_(self.entity, a=-10/embedding_dim, b=10/embedding_dim)

        # Option2: initilizing using Xavier
        self.entity = nn.Parameter(torch.FloatTensor(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.entity.data)

        # # Option3: initilizing using one-hot encoding
        # self.entity = torch.eye(num_nodes)

        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def cpu(self):
        super(PonderRelationalGraphConvModel, self).cpu() 
        self.device = torch.device('cpu')
        return self
    
    def cuda(self):
        super(PonderRelationalGraphConvModel, self).cuda() 
        self.device = torch.device('cuda')
        return self

    def forward(self, blocks):
        # forward for neighbor sampling
        h = torch.index_select(self.entity, 0, blocks[0].srcdata['id'].to(self.device))
        for layer, block in zip(self.conv_layers, blocks):
            block = block.to(self.device)
            h = layer(block, h)

        # h = self.softmax(h)
        # Lists to store $p_1 \dots p_N$ and $\hat{y}_1 \dots \hat{y}_N$
        p = []
        y = []
        
        # $\prod_{j=1}^{n-1} (1 - \lambda_j)$
        un_halted_prob = h.new_ones((h.shape[0],))

        # A vector to maintain which samples has halted computation
        halted = h.new_zeros((h.shape[0],))
        p_m = h.new_zeros((h.shape[0],))
        y_m = h.new_zeros((h.shape[0],))

        # Iterate for $N$ steps
        for n in range(1, self.max_steps + 1):
            # The halting probability $\lambda_N = 1$ for the last step
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]

            y_n = h[:, 0]

            # $$p_n = \lambda_n \prod_{j=1}^{n-1} (1 - \lambda_j)$$
            p_n = un_halted_prob * lambda_n
            # Update $\prod_{j=1}^{n-1} (1 - \lambda_j)$
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Halt based on halting probability $\lambda_n$
            halt = torch.bernoulli(lambda_n) * (1 - halted)

            # Collect $p_n$ and $\hat{y}_n$
            p.append(p_n)
            y.append(h)

            # Update $p_m$ and $\hat{y}_m$ based on what was halted at current step $n$
            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt

            # Update halted samples
            halted = halted + halt
            # Get next state $h_{n+1} = s_h(x, h_n)$
            h = torch.index_select(self.entity, 0, blocks[0].srcdata['id'].to(self.device)) # one-hot encoding
            for layer, block in zip(self.conv_layers, blocks):
                block = block.to(self.device)
                h = layer(block, h)

            # Stop the computation if all samples have halted
            if halted.sum() == h.shape[0]:
                break

        return torch.stack(y), torch.stack(p)




class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, self_loop=False, regularizer="basis", num_bases=-1, activation=None, dropout=0.2, bias=False):

        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.self_loop = self_loop
        self.num_bases = num_bases
        self.activation = activation
        self.drop_rate = dropout

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        
        if regularizer == "basis":
            if num_bases == -1:
                self.weight = nn.Parameter(torch.FloatTensor(num_rels, in_dim, out_dim))
            else:
                self.basis = nn.Parameter(torch.FloatTensor(num_bases, in_dim * out_dim))
                self.coef = nn.Parameter(torch.FloatTensor(num_rels, num_bases))
            self.message_func = self.bias_message_func
        # elif regularizer == "bdd":
        #     if in_dim % num_bases != 0 and out_dim % num_bases != 0:
        #         raise ValueError('Feature size must be a multiplier of num_bases (%d).' % self.num_bases)
        #     self.submat_in = in_dim // self.num_bases
        #     self.submat_out = out_dim // self.num_bases
        
        #     self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        #     self.message_func = self.bdd_message_func
        else:
            raise ValueError('Regularizer must be either "basis" or "bdd"')

        if self.self_loop:
            self.self_loop_weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        
        self.reset_parameters() # initialize weights

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.basis.data)
            nn.init.xavier_uniform_(self.coef.data)
        else:
            nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_loop_weight.data)
            # nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu')) # try

    def bias_message_func(self, edges):
        src = edges.src['h']
        if self.num_bases == -1:
            weight = self.weight
        else:
            weight = torch.matmul(self.coef, self.basis)
            weight = weight.reshape(self.num_rels, self.in_dim, self.out_dim)

        w = weight[edges.data['etype']]
        msg = torch.bmm(src.unsqueeze(1), w).squeeze(1)
        return {'msg': msg}

    def bdd_message_func(self, edges):
        src = edges.src['h'].view(-1, 1, self.submat_in)
        weight = self.weight[edges.data['etype']].view(-1, self.submat_in, self.submat_out)
        msg = torch.bmm(src, weight).view(-1, self.out_dim)
        return {'msg': msg}


    def forward(self, graph, in_feat):
        with graph.local_scope():
            graph.srcdata['h'] = in_feat
            graph.update_all(self.message_func, fn.mean('msg', 'h')) # aggregate using mean
            output = graph.dstdata['h']

            # apply bias and activation
            if self.bias:
                output = output + self.bias
            
            if self.self_loop:
                dst_feat = in_feat[:graph.number_of_dst_nodes()]
                output += torch.matmul(dst_feat, self.self_loop_weight)

            if self.activation:
                output = self.activation(output)
            
            output = F.dropout(output, self.drop_rate, training=self.training)
            return output

