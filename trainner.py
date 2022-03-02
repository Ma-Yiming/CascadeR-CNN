import data
import torch
import detector
import parameters


class Trainer:
    def __init__(self, net, lr, gpu=None, **kwargs):
        self.gpu = gpu
        self.net = net(**kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        if gpu is not None:
            self.net = self.net.cuda(gpu)
        self.fixed()

    def train(self, data_path, save, print_step=0):
        n = 0
        for image, target in data.data_set(data_path):
            n += 1
            if self.gpu is not None:
                image[0] = image[0].cuda(self.gpu)
                for key in target[0].keys():
                    target[0][key] = target[0][key].cuda(self.gpu)
            loss = sum(list(self.net(image, target)[1].values()))
            if print_step > 0 and n % print_step == 0:
                print("step: {}, loss: {}".format(n, loss.item()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.save(save)

    def save(self, path):
        pass

    def fixed(self):
        for param in self.net.feature.parameters():
            param.requires_grad = False


class RegionProposalNetworkTrainer(Trainer):
    def save(self, path):
        torch.save(self.net.rpn.state_dict(), path)


class FasterRCNNTrainer(Trainer):
    def __init__(self, net, lr, gpu, number, iou, rpn=None):
        super().__init__(
            net, lr, gpu, number=number, iou=iou, rpn=rpn
        )

    def save(self, path):
        torch.save(self.net.net.state_dict(), path)

    def fixed(self):
        super().fixed()
        for param in self.net.rpn.parameters():
            param.requires_grad = False


class CascadeTrainer(Trainer):
    def __init__(
        self, net, lr, gpu, number,
        iou1, iou2, rpn=None, pkl1=None, pkl2=None
    ):
        super().__init__(
            net, lr, gpu, number=number, rpn=rpn,
            iou1=iou1, iou2=iou2, net0=pkl1, cascade=pkl2
        )

    def save(self, path):
        torch.save(self.net.cascade.state_dict(), path)

    def fixed(self):
        super().fixed()
        for net in (self.net.rpn, self.net.net):
            for param in net.parameters():
                param.requires_grad = False


class TailTrainer(Trainer):
    def __init__(
        self, net, lr, gpu, number, iou1, iou2, iou3,
        rpn=None, pkl1=None, pkl2=None, pkl3=None
    ):
        super().__init__(
            net, lr, gpu, number=number, rpn=rpn,
            iou1=iou1, iou2=iou2, iou3=iou3,
            net0=pkl1, cascade=pkl2, tail=pkl3
        )

    def save(self, path):
        torch.save(self.net.tail.state_dict(), path)

    def fixed(self):
        super().fixed()
        for net in (self.net.rpn, self.net.net, self.net.cascade):
            for param in net.parameters():
                param.requires_grad = False


def train(trainer, model, lr, save, epoch=1, print_step=0, gpu=None, **kwargs):
    trainer = trainer(model, lr, gpu, **kwargs)
    while epoch:
        epoch -= 1
        trainer.train("data/train", save, print_step)


if __name__ == "__main__":
    train(
        save="parameters/rpn.pkl",
        model=detector.Head,
        trainer=RegionProposalNetworkTrainer,
        lr=1e-4, epoch=10, print_step=100, gpu=0
    )
    train(
        save="parameters/net.pkl",
        trainer=FasterRCNNTrainer,
        model=detector.FasterRCNN,
        lr=1e-4, epoch=10, print_step=100, gpu=0,
        number=49, iou=0.5, rpn="parameters/rpn.pkl"
    )
    parameters.update("parameters/net.pkl", n=4)
    train(
        save="parameters/cascade.pkl",
        trainer=CascadeTrainer,
        model=detector.CascadeNet,
        lr=1e-4, epoch=10, print_step=100, gpu=0,
        number=49, iou1=0.5, iou2=0.6,
        rpn="parameters/rpn.pkl", pkl1="parameters/net.pkl"
    )
    parameters.update("parameters/cascade.pkl", n=4)
    train(
        save="parameters/tail.pkl",
        trainer=TailTrainer,
        model=detector.CascadeRCNN,
        lr=1e-4, epoch=10, print_step=100, gpu=0,
        number=49, iou1=0.5, iou2=0.6, iou3=0.7,
        rpn="parameters/rpn.pkl",
        pkl1="parameters/net.pkl",
        pkl2="parameters/cascade.pkl"
    )
    parameters.update("parameters/tail.pkl", n=4)
    print("Over!")
