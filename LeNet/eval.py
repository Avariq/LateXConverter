import torch
from time import time
from train import LightningModel, AIDADataset, DataModule, PyTorchLeNet5
from torchvision import transforms


def main():
    data_module = DataModule(data_path='../dataset/AIDA2', limit=100)
    data_module.prepare_data()
    data_module.setup()

    model = LightningModel.load_from_checkpoint("../checkpoint/res.ckpt")
    model.eval()
    t2p = transforms.ToPILImage()
    for img, lab in data_module.test:
        t2p(img).show()
        image_nchw = img.unsqueeze(0)
        print(image_nchw.shape)
        st = time()
        with torch.no_grad():  # since we don't need to backprop
            logits = model(image_nchw)

            probas = torch.softmax(logits, dim=1)
            print(probas)
            predicted_label = torch.argmax(probas)

        print(time() - st)
        print(f'Predicted label: {predicted_label}')
        print(f'Actual label: {lab}')
        print(f'Class-membership probability {probas[0][predicted_label] * 100:.2f}%')
        input()


if __name__ == '__main__':
    main()
