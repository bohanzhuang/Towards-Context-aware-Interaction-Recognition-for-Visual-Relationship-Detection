import torch
import torch.nn as nn
from torch.autograd import Variable

class context_aware(nn.Module):

    def __init__(self, feature_dim, rank_dim, classes):
        super(context_aware, self).__init__()
        self.classifier = nn.Parameter(torch.rand(feature_dim, classes).cuda())
        self.context=  nn.Parameter(torch.rand(rank_dim, feature_dim, classes).cuda())
        self.classes = classes
        self.feature_dim = feature_dim
        self.rank_dim = rank_dim 

    def forward(self, input): 
        batch_size = input.size(0)
		# expand the dimension
        input_1 = input.unsqueeze(-1).expand(batch_size, self.rank_dim, self.feature_dim)
        expand_input = input_1.unsqueeze(-1).expand(batch_size, self.rank_dim, self.feature_dim, self.classes)

        expand_context = self.context.unsqueeze(0).expand(batch_size, self.rank_dim, self.feature_dim, self.classes)
        context = torch.sum(expand_input * expand_context, dim=1)
        context_aware_cls = self.classifier.unsqueeze(0).expand(batch_size, self.feature_dim, self.classes) + context
        return context_aware_cls
        

# low-rank the language models
class low_rank_layer(nn.Module):

    def __init__(self, language_dim, rank_dim):
        super(low_rank_layer, self).__init__()
        self.linear = nn.Linear(language_dim, rank_dim)

    def forward(self, x):
        output = self.linear(x)
        return output


# classifier layer
class classifier_layer(nn.Module):
    def __init__(self, classes):
        super(classifier_layer, self).__init__()
        self.bias = nn.Parameter(torch.rand(1, classes).cuda())
        self.classes = classes

    def forward(self, x1, x2):
    	#x1_pooling = torch.mean(torch.mean(x1, dim=3), dim=2)
    	#x1_pooling_expand = x1_pooling.unsqueeze(-1).expand(x1_pooling.size(0), x1_pooling.size(1), self.classes)
        #output = torch.sum(x1_pooling_expand * x2, dim=1) + self.bias
        output = x1 * x2
        output_2 = torch.sum(output, dim=1) + self.bias
        return output_2



class attention_layer(nn.Module):
    def __init__(self, classes):
        super(attention_layer, self).__init__()
        self.classes = classes

    def forward(self, input0, input1):
        tile_input0 = input0.unsqueeze(-1).expand(input0.size(0), input0.size(1), input0.size(2), input0.size(3), self.classes)
	tile_input1_1 = input1.unsqueeze(2).expand(input1.size(0), input1.size(1), input0.size(2), input1.size(-1))
	tile_input1_2 = tile_input1_1.unsqueeze(3).expand(tile_input1_1.size(0), tile_input1_1.size(1),tile_input1_1.size(2), input0.size(2), tile_input1_1.size(-1))
        attention_map = torch.sum(tile_input0 * tile_input1_2, dim=1)
        attention_numerator = nn.Softplus()(attention_map) + Variable(torch.FloatTensor([0.1]).cuda().expand(attention_map.size()))
        attention_dominator = torch.sum(torch.sum(attention_numerator, dim=2), dim=1)
        attention_final = attention_numerator / attention_dominator.unsqueeze(1).unsqueeze(2)               
        output = torch.sum(torch.sum(attention_final.unsqueeze(1) * tile_input0, dim=3), dim=2)
        return output












