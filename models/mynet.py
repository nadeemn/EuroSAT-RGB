import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from eurosatms_dataset import load_and_preprocess

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            *list(base_model.children())[:-1]
        )

    def forward(self, x):
        x = self.features(x)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return x

class MyNet(nn.Module):
    def __init__(self, num_classes, pretrained_model1):
        super(MyNet, self).__init__()

        self.feature_extractor_1 = FeatureExtractor(pretrained_model1)
        self.feature_extractor_2 = self.feature_extractor_1

        feature_dim = 512

        self.classifier = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:6, :, :]

        feature1 = self.feature_extractor_1(x1)
        feature2 = self.feature_extractor_2(x2)

        fused_feature = torch.cat((feature1, feature2 ), dim=1)

        output = self.classifier(fused_feature)
        return output

if __name__ == "__main__":
    sample_path = r'D:\EuroSAT_MS\EuroSAT_MS\AnnualCrop\AnnualCrop_1.tif'
    single_image = load_and_preprocess(sample_path)

    base_model = resnet18(weights = ResNet18_Weights.DEFAULT)

    model = MyNet(10, base_model)

    output = model(single_image)
