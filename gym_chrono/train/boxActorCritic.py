import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=768):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        for key, space in observation_space.spaces.items():
            if key == "image":
                image_features_dim = 10
                extractors[key] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    # This linear layer expects the flattened conv output.
                    # Adjust the hardcoded value (272384) if your input size changes.
                    nn.Linear(3136, image_features_dim),
                    nn.ReLU()
                )
                total_concat_size += image_features_dim

            elif key == "depth":
                depth_features_dim = 10
                extractors[key] = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    # Use adaptive pooling to reduce the spatial dimensions to (10,10)
                    nn.AdaptiveAvgPool2d((10, 10)),
                    nn.Flatten(),
                    # Now the flattened size is 32 * 10 * 10 = 3200
                    nn.Linear(32 * 10 * 10, depth_features_dim),
                    nn.ReLU()
                )
                total_concat_size += depth_features_dim

            else:
                # For the "data" key which is a flat vector (expected shape: (4,))
                data_features_dim = 10
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(space.shape[0], data_features_dim),
                    nn.ReLU()
                )
                total_concat_size += data_features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            x = observations[key]
            # For the "depth" key, add a channel dimension if necessary.
            if key == "depth" and x.ndim == 3:
                x = x.unsqueeze(1)
            encoded_tensor_list.append(extractor(x))
        return th.cat(encoded_tensor_list, dim=1)
