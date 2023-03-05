import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms

# setting training parameters
parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='CUDA training')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--gamma', type=int, default=0.9, metavar='N',
                    help='decay rate for scheduler')
args = parser.parse_args(args=[])

# judge cuda is available or not
cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# get the loader
def dataloader(batchsize=args.batch_size):
    train_data = datasets.FashionMNIST('data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(), ]))
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    # compute average mean and std
    n_samples_seen = 0.
    mean = 0
    std = 0
    for train_batch, train_target in train_loader:
        batch_size = train_batch.shape[0]
        train_batch = train_batch.view(batch_size, -1).to(device)
        this_mean = torch.mean(train_batch, dim=1).to(device)
        this_std = torch.sqrt(
            torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1)).to(device)
        mean += torch.sum(this_mean, dim=0)
        std += torch.sum(this_std, dim=0)
        n_samples_seen += batch_size
    mean /= n_samples_seen
    std /= n_samples_seen
    print(mean, std)
    # normalize
    train_data = datasets.FashionMNIST('data', train=True, download=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean.view(1),
                                                                std=std.view(1))]))
    test_data = datasets.FashionMNIST('data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean.view(1),
                                                               std=std.view(1))]))
    train, valid = random_split(train_data, [50000, 10000])
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    validation_loader = DataLoader(valid, batch_size=batchsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False)
    return train_loader, validation_loader, test_loader


class ResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet, self).__init__()
        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet50(pretrained=True)
        # Change the input layer to take Grayscale image, instead of RGB images.
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


class cnn4(nn.Module):
    def __init__(self):
        super(cnn4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(7 * 7 * 80, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout_2d(F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2))
        x = self.conv3(x)
        x = self.dropout_2d(F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2))
        x = x.view(-1, 7 * 7 * 80)  # flatten / reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class cnn2(nn.Module):
    def __init__(self):
        super(cnn2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3), padding=1)
        self.dropout_2d = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(7 * 7 * 20, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout_2d(F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2))
        x = self.dropout_2d(F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 7 * 7 * 20)  # flatten / reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    # epsilon hat, p = 2
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # prevent / 0

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    # update weight : wt+1 = wt - eta * g
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    # caculate 2 norm of gradient Ls(w)
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                (1.0 * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def adv_attack(model, X, y, device, eps):
    # use projected gradient descent in this project
    with torch.enable_grad():
        eps = eps
        nb_iter = 20  # num of iterations
        labels = y.to(device)
        loss_func = nn.CrossEntropyLoss()
        x_tmp = X + torch.Tensor(np.random.uniform(-eps, eps, X.shape)).type_as(X)
        pertubation = torch.zeros(X.shape).type_as(X).to(device)
        for i in range(nb_iter):
            x1 = x_tmp
            tmp_pert = pertubation
            adv_x = x1 + tmp_pert
            # get the gradient of x
            adv_x = Variable(adv_x)
            adv_x.requires_grad = True
            preds = model(adv_x)
            loss = loss_func(preds, labels)
            model.zero_grad()
            # get the gradient of the model
            grad = torch.autograd.grad(loss, adv_x)[0]
            pertubation = eps * torch.sign(grad)

        pertubation = tmp_pert
        adv_x = X + pertubation
        adv_x = adv_x.detach_()
        # clip the value into 0-1
        adv_x = torch.clamp(adv_x, min=0.0, max=1.0)
    return adv_x


def train(model, optimizer, train_loader, eps=0, adversarial=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if adversarial == True:
            data = adv_attack(model, data, target, device, eps=eps)
        output = model(data)
        loss = F.cross_entropy(output, target)
        # first backward
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward-backward pass
        F.cross_entropy(model(data), target).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)


# predict function
def eval_test(model, device, test_loader, adv=False, eps=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if adv == True:
                data = adv_attack(model, data, target, device, eps=eps)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            # sum up batch loss
            _, pred = output.data.max(dim=1)
            # get the index of the max log-probability
            correct += torch.sum(pred == target.data.long()).item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = float(correct) / len(test_loader.dataset)
    return test_loss, test_accuracy


# main function, train the dataset and print train loss, test loss for each epoch
def train_model(model, lr, train_loader, validation_loader, test_loader, train_eps=0.15, test_eps=0.1,
                adversarial=False):
    if model == 'cnn4':
        model = cnn4().to(device)
    elif model == 'cnn2':
        model = cnn2().to(device)
    else:
        # use resnet to train
        model = ResNet().to(device)
    model.eval()
    base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    min_loss = 200000
    for epoch in range(args.epochs):
        # training
        if adversarial == True:
            # adversarial training
            train(model, optimizer, train_loader, eps=train_eps, adversarial=True)
        else:
            # general training
            train(model, optimizer, train_loader, adversarial=False)
        # calculate training loss and accuracy
        trainloss, trainacc = eval_test(model, device, train_loader, adv=adversarial, eps=test_eps)
        # print trainloss and train accuracy
        print('epoch: {} train_loss: {:.4f}, train_accuracy: {:.2f}%, '.format((epoch + 1), trainloss, 100. * trainacc),
              end='')
        # calculate validation loss and accuracy
        validateloss, validateacc = eval_test(model, device, validation_loader, adv=adversarial, eps=test_eps)
        # print validation loss and accuracy
        print('validate_loss: {:.4f}, validate_accuracy: {:.2f}%, '.format(validateloss, 100. * validateacc), end='')
        # calculate test loss and accuracy
        testloss_raw, test_acc_raw = eval_test(model, device, test_loader, adv=adversarial)
        # calculate adversarial loss and  accuracy
        if adversarial == True:
            print('test_accuracy in raw data: {:.2f}%, '.format(100. * test_acc_raw), end='')
            testloss_adv, test_acc_adv = eval_test(model, device, test_loader, adv=adversarial, eps=test_eps)
            print('test_accuracy in adv data: {:.2f}%\n'.format(100. * test_acc_adv), end='')
        else:
            print('test_accuracy in raw data: {:.2f}%\n'.format(100. * test_acc_raw), end='')
        scheduler.step()
        # early stop
        if (min_loss > validateloss):
            # update globlal minimum
            min_loss = min(validateloss, min_loss)
            # save the model
            # torch.save(model.state_dict(), str(model) + '.pt')
        else:
            if ((min_loss / validateloss) < 0.9):
                break
    return model


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = dataloader()
    normal_model = ResNet().to(device)
    normal_model.load_state_dict(torch.load('normal_training.pt'))
    test_loss, test_accuracy = eval_test(normal_model, device, test_loader, adv=False, eps=0)
    print('Test accuracy in raw data under normal training: {:.2f}%\n'.format(100. * test_accuracy), end='')
    print("----------------------------------")
    adversarial_model = cnn4().to(device)
    adversarial_model.load_state_dict(torch.load('adversarial_training.pt'))
    test_loss, test_accuracy = eval_test(adversarial_model, device, test_loader, adv=False, eps=0)
    print('Test accuracy in raw data under adversarial training: {:.2f}%\n'.format(100. * test_accuracy), end='')
    test_loss, test_accuracy = eval_test(adversarial_model, device, test_loader, adv=True, eps=0.1)
    print('Test accuracy in adversarial data(eps = 0.1) under adversarial training: {:.2f}%\n'.format(100. * test_accuracy), end='')
