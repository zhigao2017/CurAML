import torch.nn as nn

from .models import register


component=16
divide=0.1

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

            x_i=torch.index_select(x,1,torch.arange(i*self.component_channel,(i+1)*self.component_channel).to(x.device))

            new_batch, new_channel, new_h, new_w=x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]
            x_i=torch.transpose(x_i,1,3).contiguous().view(new_batch*new_w*new_h,new_channel)
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



class conv_block(nn.Module):
    def __init__(self, inplanes, planes, num):
        super().__init__()


        self.relu = nn.ReLU()
        self.num=num

        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(2)

        self.exp=MetaExpLayer(planes,component)

        if self.num!=4:
            self.log=MetaLogLayer(planes,component)
        else:
            self.log=MetaLog0Layer(planes,component)


    def forward(self, x, curvature):

        out=x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.exp(out,curvature)
        out = self.log(out,curvature)

        return out





@register('convnet4')
class ConvNet4(nn.Module):

    #def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=1600):
        super().__init__()


        self.inplanes = 3

        self.curvature_1= nn.Parameter(torch.ones(component)*divide)
        self.curvature_2= nn.Parameter(torch.ones(component)*divide)
        self.curvature_3= nn.Parameter(torch.ones(component)*divide)
        self.curvature_4= nn.Parameter(torch.ones(component)*divide)
        #self.curvature_5= nn.Parameter(torch.ones(component)*divide)

        self.layer1 = conv_block(x_dim, hid_dim,1)
        self.layer2 = conv_block(hid_dim,hid_dim,2)
        self.layer3 = conv_block(hid_dim,hid_dim,3)
        self.layer4 = conv_block(hid_dim,z_dim,4)

        self.out_dim = z_dim

    def forward(self, x):

        x = self.layer1(x,self.curvature_1)
        x = self.layer2(x,self.curvature_2)
        x = self.layer3(x,self.curvature_3)
        x = self.layer4(x,self.curvature_4)

        x = nn.MaxPool2d(5)(x)
        #-----------------------------
        return x.view(x.shape[0], -1)

