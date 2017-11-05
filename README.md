# An Introduction to Deep Learning for the Physical Layer

An usable PyTorch implementation of the noisy autoencoder infrastructure in the paper "An Introduction to Deep Learning for the Physical Layer" by [Kenta Iwasaki](https://github.com/iwasaki-kenta) on behalf of Gram.AI.

Overall a fun experiment for constructing a communications system for the physical layer with transmitters/receivers in which the transmitter efficiently encodes a signal in a way such that the receiver can still, with minimal error, decode this encoded signal despite being inflicted with noise in amidst transmission.

The signal dimension for the encoded message is set to be 4, with the compressed signal representation's channel size being 2 (log_2(signal_dim)) to maximize information/bit as a basis to the principles of shannon entropy.

The signal-to-noise ratio simulated in amidst training is 7dbW. That may be changed accordingly to your preferences.

Checks for the bit error rate have been ignored for the decoder, and instead the reconstruction of the input based on categorical cross-entropy is used to validate model generalization and performance.

Training for the model is done using [TorchNet](https://github.com/pytorch/tnt).

## Description

> We present and discuss several novel applications
of deep learning (DL) for the physical layer. By interpreting
a communications system as an autoencoder, we develop a
fundamental new way to think about communications system
design as an end-to-end reconstruction task that seeks to jointly
optimize transmitter and receiver components in a single process.
We show how this idea can be extended to networks of multiple
transmitters and receivers and present the concept of radio
transformer networks (RTNs) as a means to incorporate expert
domain knowledge in the machine learning (ML) model. Lastly,
we demonstrate the application of convolutional neural networks
(CNNs) on raw IQ samples for modulation classification which
achieves competitive accuracy with respect to traditional schemes
relying on expert features. The paper is concluded with a
discussion of open challenges and areas for future investigation.

Paper written by Tim O'Shea and Jakob Hoydis. For more information, please check out the paper [here](https://arxiv.org/abs/1702.00832).

## Requirements

* Python 3
* PyTorch
* TorchNet
* TQDM

## Usage


**Step 1** Start training.

```console
$ python3 radio_transformer_networks.py
```

**Step 2** Call `model.decode_signal(x)` on any noisy data on the transmitter's end.

## Benchmarks

Achieves 100% within a span of ~30 epochs.

Default PyTorch Adam optimizer hyperparameters were used with no learning rate scheduling. Epochs with batch size of 256 takes half a second on a Razer Blade w/ GTX 1050. 

## TODO

* Signal modulation classification using convolutional neural networks as outlined on the paper.

## Contact/Support

Gram.AI is currently heavily developing a wide number of AI models to be either open-sourced or released for free to the community, hence why we cannot guarantee complete support for this work.

If any issues come up with the usage of this implementation however, or if you would like to contribute in any way, please feel free to send an e-mail to [kenta@gram.ai](kenta@gram.ai) or open a new GitHub issue on this repository.