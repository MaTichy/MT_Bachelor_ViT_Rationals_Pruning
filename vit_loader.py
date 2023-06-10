from efficientViT import ViT
from simpleViT import simple_ViT
from linformer import Linformer


def vit_loader(args):
    if(args == "simple"):
        
        model = simple_ViT(
            image_size = 32, # 32, 64 svhn / 64 tiny ;Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
            patch_size = 4, # 4 for 32, 16 for 224 svhn / 8 tiny, ;Number of patches. image_size must be divisible by patch_size. The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
            num_classes = 10, #10, 200 Number of classes to classify.
            dim = 1024, #1024, Last dimension of output tensor after linear transformation nn.Linear(..., dim).
            depth = 6, # 6 Number of Transformer blocks. 9
            heads = 12, # 16 Number of heads in Multi-head Attention layer. 12
            mlp_dim = 2048, # 2048 Dimension of the MLP (FeedForward) layer.
            lr=4.365158322401661e-05
        )
    
    elif(args == "efficient"):
        """
        An implementation of Linformer in Pytorch. Linformer comes with two deficiencies. (1) It does not work for the auto-regressive case. (2) Assumes a fixed sequence length. However, if benchmarks show it to perform well enough, it will be added to this repository as a self-attention layer to be used in the encoder.

        Linformer has been put into production by Facebook!
        """
        
        efficient_transformer = Linformer(
            dim=256, #128 256
            seq_len=64+1,  # 8x8 patches + 1 cls-token 64+1 256+1
            depth=12,
            heads=16, #8 16
            k=64 #64 128 input length
        )

        model = ViT(
            dim=256, # 128 256
            image_size=3232, # 64 tiny / 32 svhn 224
            patch_size=4, # 8 tiny / 4 svhn , 2
            num_classes=10, # 200 tiny / 10 svhn
            transformer=efficient_transformer, # efficient_transformer
            channels=3, # 3
            lr=0.001445439770745928
        )
    return model   
            
