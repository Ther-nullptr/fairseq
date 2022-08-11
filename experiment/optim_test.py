import torch
import torch.nn

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear1(x)

if __name__ == '__main__':
    net = Net()
    main_model_params = [{
        "params": [p for _, p in net.named_parameters()],
        "weight_decay": 0.0,
        "lr": -0.1
    }]
    optimizer = torch.optim.AdamW(
                main_model_params,
            )
    print(optimizer)