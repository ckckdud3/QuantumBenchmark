import torch
from torch.utils.data.dataloader import DataLoader
import pickle
import logging
import time
import matplotlib.pyplot as plt
import torch.nn as nn

from dataset.datasets import get_pca_dataset

data_dim = 12
train_size = 2000
test_size = 500
batch_size = 500
epoch = 50
save_to = 'normal_net'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler(f'{save_to}.log', mode='w')
                    ])
logger = logging.getLogger('main')

class Net(nn.Module):

    def __init__(self, data_dim, hidden_dim, out_dim):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(data_dim, hidden_dim, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=torch.float64)
        self.relu = nn.ReLU()

    
    def forward(self, data):

        logits = self.fc1(data)
        logits = self.relu(logits)
        return self.fc2(logits)


net = Net(12, 6, 2).to(device)

train_dataset, test_dataset = get_pca_dataset(data_dim, train_size, test_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)


criterion = nn.CrossEntropyLoss()

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

outdata = {}

cost_data = []
accuracy_per_epoch = []


start = time.time()
logger.info(f'Training started : Batch size = {batch_size}, epoch = {epoch}')

net.train()

for e in range(epoch):

    logger.info(f'Epoch #{e+1}')

    for idx, (data, label) in enumerate(train_loader):
        
        opt.zero_grad()

        data, label = data.to(device), label.to(device)
        logger.info(f'Iteration #{idx+1}')
        logits = net(data)
        cost = criterion(logits, label)
        cost.backward()

        logger.info(f'Cost : {cost}')
        cost_data.append(cost.item())
        opt.step()

    logger.info(f'Epoch #{e+1} Test')

    count = 0
    for idx, (data, label) in enumerate(test_loader):

        data, label = data.to(device), label.to(device)
        eval = net(data)
        count += torch.sum(torch.max(eval,axis=1).indices == label).item()

    logger.info(f'Epoch #{e+1} Accuracy : {count/test_size}')
    accuracy_per_epoch.append(count/test_size)

end = time.time()

seconds = int(end - start)
modseconds = (seconds) % 60
minutes = (seconds // 60) % 60
hours = (seconds // 3600) % 24
days = (seconds // 86400)

logger.info(f'Training ended. Elapsed time: {days} days {hours} hours {minutes} minutes {modseconds} seconds')

outdata['elapsed_time'] = seconds
outdata['num_epoch'] = epoch
outdata['batch_size'] = batch_size
outdata['cost_data'] = cost_data
outdata['accuracy_per_epoch'] = accuracy_per_epoch

logger.info('Test started')

net.eval()

count = 0
for idx, (data, label) in enumerate(test_loader):

    data, label = data.to(device), label.to(device)
    eval = net(data)
    count += torch.sum(torch.max(eval,axis=1).indices == label).item()


logger.info(f'Accuracy : {count/test_size}')
accuracy_per_epoch.append(count/test_size)

outdata['test_accuracy'] = (count/test_size)

with open(save_to+'_traindata.pickle', 'wb') as f:
    pickle.dump(outdata, f)
    f.close()