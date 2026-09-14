import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TeacherModel(nn.Module): # Модель учителя, принимает изображение + координаты цели
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.target = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)

        )
    def forward(self, image, target):
        x = self.backbone(image)              
        x = x.flatten(1)                
        # Признаки цели
        g = self.target(target)         
        # Объединяем
        combined = torch.cat([x, g], dim=1)
        return self.head(combined)



class StudentModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # 42x42
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 21x21
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 11x11
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (batch, 64, 1, 1)

        self.target_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image, target):
        x = self.vision_layer(image)      # (batch, 64, 11, 11)
        x = self.global_pool(x)           # (batch, 64, 1, 1)
        x = x.flatten(1)                  # (batch, 64)
        g = self.target_mlp(target)       # (batch, 16)
        combined = torch.cat([x, g], dim=1)  # (batch, 80)
        return self.head(combined)        # (batch, 2)
    
class StudentModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image):
        x = self.vision_layer(image)   # (batch, 64, 11, 11)
        x = self.global_pool(x)        # (batch, 64, 1, 1)
        x = x.flatten(1)               # (batch, 64)
        return self.head(x)            # (batch, 2)