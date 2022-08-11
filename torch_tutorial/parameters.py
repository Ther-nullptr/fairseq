import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear2 = torch.nn.Linear(3,4)
    def forward(self,x):
        pass

class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear3 = torch.nn.Linear(3,4)
    def forward(self,x):
        pass

if __name__ == '__main__':
    net = Net()
    # torch.save(net.state_dict(), 'tmp.pt')
    # print(net.state_dict())
    # net1 = Net2()
    # print(net1.state_dict())
    # params = torch.load('tmp.pt')
    # net1.load_state_dict(params, strict=False)
    # print(net1.state_dict())
    state_dict = net.state_dict()
    print(state_dict)
    del state_dict['linear1.bias']
    print(state_dict)