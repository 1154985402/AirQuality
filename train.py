import torch
import torch.optim
import torch.utils.data
import GA
import RBFnet.RBFNet as RBFnet
import data.DataLoader
import RBFnet.RBF as rbf
import torch.nn.functional as F
from util import plot_data
from util import plot_relation

data_path = 'data/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv'
basis_func = rbf.gaussian
dataset = data.DataLoader.Datalder(data_path)


def train(model,  epochs, batch_size, lr, loss_func ,data_path, idx):
    print("training model...")
    model.train()

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    # gamma = 0.95
    # scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma)
    epoch = 0
    training_loss = []
    epoch_arr = []
    validating_loss  = []
    while epoch < epochs:
        epoch += 1
        losses = []
        for x_batch, y_batch in trainloader:
            batches = 0
            progress = 0
            x_batch = x_batch[:,idx]
            batches += 1
            optimiser.zero_grad()
            y_hat = model.forward(x_batch).squeeze(-1)
            current_loss = loss_func(y_hat, y_batch)
            current_loss.backward()
            optimiser.step()
            # scheduler_lr.step()
            losses.append(current_loss)

        loss_m = torch.mean(torch.tensor(losses ,dtype=torch.float)).item()
        validloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=True)
        valiloss = []
        for x_vali , y_vali  in validloader:
            batches = 0
            progress = 0
            x_vali = x_vali[:,idx]
            batches += 1
            y_hat = model.forward(x_vali).squeeze(-1)
            current_loss = loss_func(y_hat, y_vali)
            valiloss.append(current_loss)

        loss_v = torch.mean(torch.tensor(valiloss , dtype=torch.float)).item()
        print('Epoch %d :   training  Loss: %0.6f   |    '
              '  testing  Loss: %0.6f' % (epoch,  loss_m , loss_v))
        training_loss.append(loss_m)
        epoch_arr.append(epoch)
        validating_loss.append(loss_v)
    plot_data(epoch_arr, training_loss , validating_loss)
    torch.save(model,'model/RBFnet.pth')



def predict(model ,x ,y, idx):
    x = x[:, idx]
    y  =  y.detach().numpy()
    y_hat = model.forward(x).squeeze(-1)
    y_hat = y_hat.detach().numpy()
    plot_relation(y_hat , y )

best_ind = []

if __name__ == '__main__':
    data_path = 'data/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv'
    best_ind= GA.get_best()
    epochs = 50
    batch_size = 2000
    lr = 0.006
    loss_func = F.mse_loss
    idx = [i for i, x in enumerate(best_ind) if x == 1]
    model = RBFnet.Network(len(idx), 25, basis_func)
    train(model , epochs , batch_size , lr ,loss_func, data_path, idx)

    datas = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
    for i, data in enumerate(datas):
        pass

    Xdata = data[0]
    ydata = data[1]
    predict(model,Xdata,ydata ,idx)
