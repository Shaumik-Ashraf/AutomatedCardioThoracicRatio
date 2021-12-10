# Models

| Name | Model | Hyperparams |
| resUNet | UNet | input-dim = 1, output-dim = 3, pretrained resnet encoder |
| UNet2 | UNet | input-dim = 1, output-dim = 3 |
| resnet | resnet18 | input-dim = 1, output-dim = 3|
| vit  | Google's ViT | image-size = 512, patch-size=32, num-classes=1, dim=1024, depth = 4, heads = 8, mlp_dim = 512, dropout = 0.1, emb-dropout = 0.1 |
| vit-sm | Google's ViT | image-size = 512, patch-size=32, num-classes=1, dim=1024, depth = 4, heads = 4, mlp_dim = 64, dropout = 0.1, emb-dropout = 0.1 |
| vit-lg | Google's ViT | image-size = 512, patch-size=32, num-classes=1, dim=1024, depth = 16, heads = 16, mlp_dim = 1024, dropout = 0.25, emb-dropout = 0.25 |

vit_take1 was just another run of vit

num-classes is 1 for transformers because they are preforming regression instead of classification

*_xval files are from cross validation training

No models were pretrained unless otherwise specified.
