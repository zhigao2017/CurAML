import torch.nn as nn
import torch
from .models import register
from hyptorch.pmath import dist_matrix, expmap0, logmap0, RiemannianBatchNorm, expmap, logmap,lambda_x

component=16
divide=-1

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class MetaLogLayer(nn.Module):
    def __init__(self, in_channels,  component_num):

        super(MetaLogLayer, self).__init__()

        self.component_num=component_num
        self.in_channels=in_channels
        self.component_channel=int (in_channels/component_num)

        self.location = nn.Parameter(torch.zeros(self.component_num,self.component_channel))
        #nn.init.xavier_uniform_(self.location)


    def forward(self, x, curvature):


        out=[]
        for i in range(self.component_num):                         
            location_i=expmap0(self.location[i],k=curvature[i]).unsqueeze(0)

            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)
            x_i=logmap(location_i, x_i,k=curvature[i])
            out.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)
            #out.append(x_i)

        out = torch.cat(out,dim=1) # (n,c,h,w)
        return out

class MetaExpLayer(nn.Module):
    def __init__(self, in_channels,  component_num):

        super(MetaExpLayer, self).__init__()

        self.component_num=component_num
        self.in_channels=in_channels
        self.component_channel=int(in_channels/component_num)

        self.location = nn.Parameter(torch.zeros(self.component_num,self.component_channel))
        #nn.init.xavier_uniform_(self.location)


    def forward(self, x, curvature):

        out=[]
        for i in range(self.component_num):

            location_i=expmap0(self.location[i],k=curvature[i]).unsqueeze(0)

            # location_i_norm=torch.norm(self.location[i], dim=-1, keepdim=True)          
            # location_i=expmap0(self.location[i]/location_i_norm,k=curvature[i]).unsqueeze(0)

            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)

            # x_i_norm=torch.norm(x_i,dim=-1,keepdim=True)
            # x_i=x_i / x_i_norm

            # print('-----------------------')
            # print ('x_i 1', x_i)
            # print ('location_i',location_i)
            if curvature[i]>0:
                # print('torch.sum(x_i*location_i)',torch.sum(x_i*location_i))
                x_i=x_i-location_i*torch.sum(x_i*location_i)
            else:
                # print('lambda',lambda_x(location_i,curvature[i]))
                x_i=(1/(lambda_x(location_i,curvature[i])*lambda_x(location_i,curvature[i])))*x_i
            # print ('x_i 2', x_i)

            # x_i_norm=torch.norm(x_i,dim=-1,keepdim=True)
            # x_i=x_i / x_i_norm

            x_i=expmap(location_i,x_i,k=curvature[i])
            out.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)
            #out.append(x_i)

        out = torch.cat(out,dim=1) # (n,c,h,w)    

        return out

class MetaLog0Layer(nn.Module):
    def __init__(self, in_channels,  component_num):

        super(MetaLog0Layer, self).__init__()

        self.component_num=component_num
        self.in_channels=in_channels
        self.component_channel=int(in_channels/component_num)
        #self.curvature= nn.Parameter(torch.ones(component)*divide)

    def forward(self, x, curvature):
        out=[]
        for i in range(self.component_num):                         

            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)
            x_i=logmap0(x_i,k=curvature[i])
            out.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)
            #out.append(x_i)

        out = torch.cat(out,dim=1) # (n,c,h,w)
        return out




class MetaProjection(nn.Module):
    def __init__(self, in_channels,  component_num):
        super(MetaProjection, self).__init__()
        self.component_num=component_num
        self.in_channels=in_channels
        self.component_channel=int(in_channels/component_num)
    def forward(self, x, curvature):
        out=[]
        for i in range (self.component_num):

            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)
            x_i=logmap0(x_i,k=curvature[i])
            out.append ( torch.transpose(x_i.contiguous().view(new_batch,new_w,new_h,new_channel),1,3) )# (batch,channel,h,w)
            #out.append(x_i)

        out = torch.cat(out,dim=1) # (n,c,h,w)
        return out





class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, num):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)
        self.num=num

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

        self.exp=MetaExpLayer(planes,component)

        if self.num!=4:
            self.log=MetaLogLayer(planes,component)
        else:
            self.log=MetaLog0Layer(planes,component)

        #self.curvature= nn.Parameter(torch.ones(component)*divide)

    def forward(self, x, curvature):

        out=x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.exp(out,curvature)


        out = self.log(out,curvature)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3

        self.curvature_1= nn.Parameter(torch.ones(component)*divide)
        self.curvature_2= nn.Parameter(torch.ones(component)*divide)
        self.curvature_3= nn.Parameter(torch.ones(component)*divide)
        self.curvature_4= nn.Parameter(torch.ones(component)*divide)
        #self.curvature_5= nn.Parameter(torch.ones(component)*divide)

        self.layer1 = self._make_layer(channels[0],1)
        self.layer2 = self._make_layer(channels[1],2)
        self.layer3 = self._make_layer(channels[2],3)
        self.layer4 = self._make_layer(channels[3],4)

        self.out_dim = channels[3]

        #self.log0=MetaLog0Layer(channels[3],component)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes,num):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample,num)
        self.inplanes = planes
        return block

    def forward(self, x):
        #print('x1 shape', x.shape)
        x = self.layer1(x,self.curvature_1)
        #print('x2 shape', x.shape)
        x = self.layer2(x,self.curvature_2)
        x = self.layer3(x,self.curvature_3)
        x = self.layer4(x,self.curvature_4)
        #x = self.log0 (x,self.curvature_5)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x

    '''
    def curvature_parameters(self):
        return self.curvature
    def parameters(self):
        return self.layer1.parameters()+self.layer2.parameters()+self.layer3.parameters()+self.layer4.parameters()
    '''



@register('resnet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])


@register('resnet12-wide')
def resnet12_wide():
    return ResNet12([64, 160, 320, 640])

