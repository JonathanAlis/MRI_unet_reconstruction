import torch
from torch import nn
import torch.nn.functional as F
from .utils import get_default_device, to_device, DeviceDataLoader
    
class Unet(nn.Module):    
    def __init__(self, num_inputs=1): # imsize=[3,256,256]
        super().__init__()
        #o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        #o=[256+(2*1)-3-(2*0)]/2 +1=
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(num_inputs, 64, 3, stride=1, padding=1,dilation=1), #in: 1,256,256, out: 64,256,256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),            
            #maxpooling
            nn.Conv2d(64, 64, 3, stride=1, padding=1), #in: 64,256,256, out: 64,256,256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #maxpooling
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1), #in: 64,256,256, out: 64,128,128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),            
            #maxpooling
            nn.Conv2d(64, 128, 3, stride=1, padding=1), #in: 64,128,128, out: 128,128,128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #maxpooling
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #in:128,128,128, out: 128,128,128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #max
        )
        #input: 128,64,64
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1), #in: 128,128,128, out: 128,64,64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),            
            #maxpooling
            nn.Conv2d(128, 256, 3, stride=1, padding=1), #in: 128,64,64, out: 256,64,64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1, output_padding=0), #in: 256,64,64, out: 128,64,64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),#in: 128,64,64, out: 128,128,128
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1, output_padding=0), #in: 256,128,128, out: 128,128,128
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, output_padding=0),#in: 128,128,128, out: 64,128,128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64, 3, stride=2, padding=1, output_padding=1),#in: 64,128,128, out: 64,256,256
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, output_padding=0), #in: 128,256,256, out: 64,256,256
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=0),#in: 64,256,256 out: 64,256,256
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1, output_padding=0),#in: 64,256,256 out: 1,256,256
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.decoder_1(x3)
        x5 = self.decoder_2(torch.cat((x4,x2), 1))
        x6 = self.decoder_3(torch.cat((x5,x1), 1))

        return x6
    def name(self):
        return 'Unet'

        
                
def testUnet(gen):
  unet = Unet().float()
  print(type(gen))
  for data in iter(gen):
      x=data[0]
      print(type(data))
      print('x shape:', x.shape)
      x=x.view(-1, 1, 256,256)
      print('x shape:', x.shape)
      print(x.dtype)
      x1 = unet.encoder_1(x)
      print('x1 shape:', x1.shape)
      x2 = unet.encoder_2(x1)
      print('x2 shape:', x2.shape)
      x3 = unet.encoder_3(x2)
      print('x3 shape:', x3.shape)
      x4 = unet.decoder_1(x3)
      print('x4 shape:', x4.shape)
      print('x4 cat x2 shape:', torch.cat((x4,x2), 1).shape)      
      x5 = unet.decoder_2(torch.cat((x4,x2), 1))
      print('x5 shape:', x5.shape)
      print('x5 cat x1 shape:', torch.cat((x5,x1), 1).shape)
      x6 = unet.decoder_3(torch.cat((x5,x1), 1))
      print('x6 shape:', x6.shape)

      x
      break

    
def main():
    from torchsummary import summary

    device=get_default_device()

    unet = to_device(Unet(num_inputs=3),device)
    
    s=summary(unet,input_size=(3,256,256), device = device.__str__())
    print(s)
    print(get_default_device().__str__())
    
if __name__ == "__main__":
    main()