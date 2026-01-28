import torch
import torch.nn as nn

def encode_with_resnet(encoder, x):
    encode = []
    x = encoder.conv1(x)
    x = encoder.bn1(x)
    x = encoder.act1(x)

    x = encoder.layer1(x)
    encode.append(x)
    x = encoder.layer2(x)
    encode.append(x)
    x = encoder.layer3(x)
    encode.append(x)
    x = encoder.layer4(x)
    encode.append(x)

    return encode
