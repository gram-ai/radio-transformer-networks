import torch
from torch import nn


class RadioTransformerNetwork(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RadioTransformerNetwork, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels)
        )

    def decode_signal(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)

        # Normalization
        x = (self.in_channels ** 2) * (x / x.norm(dim=-1)[:, None])

        # 7dBW to Er/N0
        training_signal_noise_ratio = 5.01187

        # bit / channel_use
        communication_rate = 1

        x += Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5)).cuda()

        x = self.decoder(x)

        return x


if __name__ == "__main__":
    from tqdm import tqdm
    from torchnet.engine import Engine
    from torch.autograd import Variable
    from torch.optim import Adam
    import torchnet as tnt

    channel_size = 4

    model = RadioTransformerNetwork(channel_size, compressed_dim=2)
    model.cuda()

    train_labels = (torch.rand(10000) * channel_size).long()
    train_data = torch.sparse.torch.eye(channel_size).index_select(dim=0, index=train_labels)

    test_labels = (torch.rand(1500) * channel_size).long()
    test_data = torch.sparse.torch.eye(channel_size).index_select(dim=0, index=test_labels)

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(channel_size, normalized=True)

    loss_fn = nn.CrossEntropyLoss()


    def get_iterator(mode):
        data = train_data if mode else test_data
        labels = train_labels if mode else test_labels
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=256, num_workers=4, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        outputs = model(data)

        loss = loss_fn(outputs, labels)

        return loss, outputs


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        reset_meters()

        engine.test(processor, get_iterator(False))

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))


    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=100, optimizer=optimizer)


