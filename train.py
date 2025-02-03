import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_tools.utils import SPDataset, pred, metric
from Net.New_ComModel import Attention_CRF

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
def load_data(data_path, label_path, kingdom_path, aa_path, maxlen=70, test_path="./test_data/embedding/test_feature.npy"):
    dataset = SPDataset(data_path, label_path, kingdom_path, aa_path, maxlen, test_path)
    return DataLoader(dataset, batch_size=256, shuffle=True)

# 定义模型
def create_model(config, config1, cnn_configs, lstm_lan_config, lstm_config, use_CRF=False, use_attention=True, reweight_ratio=None):
    model = Attention_CRF(config, config1, cnn_configs, lstm_lan_config, lstm_config, use_CRF, use_attention, reweight_ratio)
    return model.to(device)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

# 主函数
def main():
    # 配置参数
    config = {...}  # 根据实际情况填写
    config1 = {...}  # 根据实际情况填写
    cnn_configs = [{...}, {...}]  # 根据实际情况填写
    lstm_lan_config = {...}  # 根据实际情况填写
    lstm_config = {...}  # 根据实际情况填写

    # 加载数据
    train_loader = load_data('./data/train_data.txt', './data/train_labels.txt', './data/train_kingdoms.txt', './data/train_aa.txt')

    # 创建模型
    model = create_model(config, config1, cnn_configs, lstm_lan_config, lstm_config, use_CRF=True, use_attention=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=25)

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()