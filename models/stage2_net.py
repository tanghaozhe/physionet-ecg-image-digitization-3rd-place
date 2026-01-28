import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerModel
from .backbones import encode_with_resnet
from .decoders import CoordUnetDecoder

def compute_dice_loss(inputs, targets, smooth=1.0):
    # inputs: (B, C, H, W) Logits
    # targets: (B, C, H, W) 0/1 Mask
    
    inputs_sigmoid = torch.sigmoid(inputs)
    
    B = inputs.shape[0]
    inputs_flat = inputs_sigmoid.view(B, -1)
    targets_flat = targets.view(B, -1)
    
    intersection = (inputs_flat * targets_flat).sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    
    return 1.0 - dice.mean()

class Stage2Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # Segformer encoder from Hugging Face
        self.encoder = SegformerModel.from_pretrained(
            cfg.backbone,
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        )

        # Decoder with 5 upsampling stages (1/32 -> 1/1)
        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0, 0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0, 0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )

        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # Segformer encoder
        outputs = self.encoder(x, output_hidden_states=True)
        encode = list(outputs.hidden_states)  # Convert tuple to list

        # Decoder
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None, None]
        )

        # CoordY based on actual output size
        _, _, H_out, W_out = last.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        # Pixel head
        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        # Output
        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output


class Stage2ConvNeXtV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # ConvNeXt V2 encoder with features_only
        self.encoder = timm.create_model(
            model_name=cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # Decoder with 5 upsampling stages (1/32 -> 1/1)
        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0, 0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0, 0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )

        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # ConvNeXt V2 encoder
        encode = self.encoder(x)

        # Decoder
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None, None]
        )

        # CoordY based on actual output size
        _, _, H_out, W_out = last.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        # Pixel head
        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        # Output
        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

            # bce_loss = F.binary_cross_entropy_with_logits(
            #     pixel,
            #     batch['pixel'].to(device),
            #     pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            # )
            # target_masks = batch['pixel'].to(device)
            # dice_loss = compute_dice_loss(pixel, target_masks)
            
            # output['pixel_loss'] = 0.5 * bce_loss + 0.5 * dice_loss

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output


class Stage2HRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # HRNet encoder with features_only
        self.encoder = timm.create_model(
            model_name=cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # Decoder
        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )

        # Pixel head
        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # HRNet encoder
        encode = self.encoder(x)

        # Decoder
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        # CoordY
        coordy = torch.arange(H, device=device).reshape(1, 1, H, 1).repeat(B, 1, 1, W)
        coordy = coordy / (H - 1) * 2 - 1

        # Pixel head
        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        # Output
        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output


class Stage2ResNeSt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # ResNeSt encoder
        self.encoder = timm.create_model(
            model_name=cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            num_classes=0,
            global_pool=''
        )

        # Decoder with 4 upsampling stages (1/16 -> 1/1)
        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )

        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # ResNeSt encoder with encode_with_resnet helper
        e = self.encoder
        encode = encode_with_resnet(e, x)

        # Decoder
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        # CoordY based on actual output size
        _, _, H_out, W_out = last.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        # Pixel head
        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        # Output
        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output


class Stage2EfficientNetV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # EfficientNetV2 encoder with features_only (outputs 5 stages)
        self.encoder = timm.create_model(
            model_name=cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        # Decoder with 5 upsampling stages (1/64 -> 1/2)
        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim + [cfg.decoder_dim[-1]],
                scale=[2, 2, 2, 2, 2]
            )

        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # EfficientNetV2 encoder (returns 5 feature maps)
        encode = self.encoder(x)

        # Decoder (use last feature as input, first 4 as skip connections)
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        # CoordY based on actual output size
        _, _, H_out, W_out = last.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        # Pixel head
        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        # Output
        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output


class Stage2Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_type = ['infer', 'loss']

        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        self.encoder = timm.create_model(
            model_name=cfg.backbone,
            pretrained=cfg.pretrained,
            in_chans=3,
            num_classes=0,
            global_pool=''
        )

        if cfg.use_coord_conv:
            self.decoder = CoordUnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )
        else:
            from .decoders import UnetDecoder
            self.decoder = UnetDecoder(
                in_channel=cfg.encoder_dim[-1],
                skip_channel=cfg.encoder_dim[:-1][::-1] + [0],
                out_channel=cfg.decoder_dim,
                scale=[2, 2, 2, 2]
            )

        self.pixel = nn.Conv2d(cfg.decoder_dim[-1] + 1, cfg.num_output_channels, 1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)

        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        coordy = torch.arange(H, device=device).reshape(1, 1, H, 1).repeat(B, 1, 1, W)
        coordy = coordy / (H - 1) * 2 - 1

        e = self.encoder
        encode = encode_with_resnet(e, x)

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        last = torch.cat([last, coordy], dim=1)
        pixel = self.pixel(last)

        output = {}
        if 'loss' in self.output_type:
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.cfg.pixel_pos_weight]).to(device),
            )

        if 'infer' in self.output_type:
            output['pixel'] = torch.sigmoid(pixel)

        return output
