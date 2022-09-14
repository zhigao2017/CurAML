import math

import torch
import torch.nn as nn

import models
import utils
from .models import register
from    torch.nn import functional as F

@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        #classifier_args['compoent_num'] = 16
        #classifier_args['feature_wh'] = 5
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


@register('attention-classifier')
class AttentionClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, compoent_num, feature_wh):
        super().__init__()
        self.in_dim=in_dim
        self.n_classes=n_classes
        self.compoent_num=compoent_num
        self.component_channel=int(self.in_dim/self.compoent_num)
        self.feature_wh=feature_wh
        self.feature_num=self.feature_wh*self.feature_wh

        self.prototype = nn.Parameter(torch.randn(self.compoent_num,self.n_classes,self.component_channel))

        self.att1=nn.Linear(self.feature_wh*self.feature_wh*self.compoent_num, self.feature_wh*self.feature_wh*self.compoent_num)
        self.att2=nn.Linear(self.feature_wh*self.feature_wh*self.compoent_num, self.feature_wh*self.feature_wh*self.compoent_num)

        #self.curvature= nn.Parameter(torch.ones(component)*divide)

    def forward(self, x):

        batch=x.shape[0]

        dis=torch.zeros(self.compoent_num,batch*self.feature_num,self.n_classes).to(x.device)

        for i in range (self.compoent_num):

            #c_i=self.curvature[i]
            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            #batch, channel, h, w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            #x_i=torch.transpose(x_i,1,3).contiguous().view(batch*w*h,channel)
            #x_i=logmap0(x_i,k=c_i)
            #x_i = torch.transpose(x_i.contiguous().view(batch,w,h,channel),1,3) # (batch,channel,h,w)

            x_i=torch.reshape(x_i,(batch,self.component_channel, self.feature_num))
            x_i=(torch.transpose(x_i,1,2)).contiguous().view(batch*self.feature_num, self.component_channel)
            dis[i,:,:] = torch.mm(x_i,self.prototype[i,:,:].t())


        dis=dis.view(self.compoent_num,batch,self.feature_num,self.n_classes)
        dis=dis.permute(1,3,0,2).contiguous()
        dis=dis.view(batch*self.n_classes,self.compoent_num*self.feature_num)

        attention_weight=self.att1(dis)
        attention_weight=self.att2(attention_weight)

        attention_weight=F.softmax(attention_weight)
        d=torch.sum(dis*attention_weight,dim=1)
        logit=d.view(batch,self.n_classes)

        return logit


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

