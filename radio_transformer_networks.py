import torch
from torch import nn

NUM_EPOCHS = 100
BATCH_SIZE = 256
CHANNEL_SIZE = 4
USE_CUDA = False


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

        # Normalization.
        x = (self.in_channels ** 2) * (x / x.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio = 5.01187

        # bit / channel_use
        communication_rate = 1

        # Simulated Gaussian noise.
        noise = Variable(torch.randn(*x.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        if USE_CUDA: noise = noise.cuda()
        x += noise

        x = self.decoder(x)

        return x


if __name__ == "__main__":
    from tqdm import tqdm
    from torchnet.engine import Engine
    from torch.autograd import Variable
    from torch.optim import Adam
    import torchnet as tnt
    import math

    model = RadioTransformerNetwork(CHANNEL_SIZE, compressed_dim=int(math.log2(CHANNEL_SIZE)))
    if USE_CUDA: model = model.cuda()

    train_labels = (torch.rand(10000) * CHANNEL_SIZE).long()
    train_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels)

    test_labels = (torch.rand(1500) * CHANNEL_SIZE).long()
    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(CHANNEL_SIZE, normalized=True)

    loss_fn = nn.CrossEntropyLoss()


    def get_iterator(mode):
        data = train_data if mode else test_data
        labels = train_labels if mode else test_labels
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        data = Variable(data)
        labels = Variable(labels)

        if USE_CUDA:
            data = data.cuda()
            labels = labels.cuda()

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

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
