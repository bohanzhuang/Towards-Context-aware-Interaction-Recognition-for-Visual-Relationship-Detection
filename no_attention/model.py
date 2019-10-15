import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from new_layers import context_aware, low_rank_layer, classifier_layer


class context_aware_model(nn.Module):

    def __init__(self, features, feature_dim, language_dim, rank_dim, num_classes, init_weights=False):
        super(context_aware_model, self).__init__()
        self.features = features
        self.context_aware = context_aware(feature_dim, rank_dim, num_classes)
        self.low_rank = low_rank_layer(language_dim, rank_dim)
        self.classifier = classifier_layer(num_classes)
        self.context_aware_module = nn.Sequential(self.context_aware, self.low_rank, self.classifier)        
        if init_weights:
            self._initialize_weights()

    def forward(self, x, language):
        x = self.features(x)
#        x = x.view(x.size(0), -1)
        low_rank_language = self.low_rank(language)
        model = self.context_aware(low_rank_language)
        output = self.classifier(x, model)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg_model(config, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    args = (config['feature_dim'], config['language_dim'], config['rank_dim'], config['num_classes'])
    model = context_aware_model(make_layers(cfg['D'], batch_norm=False), *args, **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model




