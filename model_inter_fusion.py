import torch
import torchvision
import torchvision.models as models
import config
# from longformer.longformer import LongformerSelfAttention

class ImageCaptioningModel(torch.nn.Module):
    def __init__(self, target_vocab_len, embedding_size=768, num_layers=12, heads=12, dropout=0.2, feedforward_dim=3072, max_len=1024):
        super().__init__()
        self.max_len = max_len

        # Image Encoder
        self.image_encoder_resnet = models.resnet101(pretrained=True)
        self.image_encoder_efficientnet = models.efficientnet_v2_s(pretrained=True)
        
        num_ftrs_resnet = self.image_encoder_resnet.fc.in_features
        num_ftrs_efficientnet = self.image_encoder_efficientnet._fc.in_features
        self.image_encoder_resnet.fc = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                    torch.nn.Linear(num_ftrs_resnet,
                                                                    embedding_size, bias=True))
        self.image_encoder_efficientnet._fc = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                    torch.nn.Linear(num_ftrs_efficientnet,
                                                                    embedding_size, bias=True))
        
        # Improved Attention
        self.attention =  torch.nn.MultiheadAttention(embed_dim=embedding_size, num_heads=heads)

        # Decoder
        self.decoder = config.PRETRAINED_GPT2MODEL
        self.decoder.resize_token_embeddings(target_vocab_len)

        # Reinforcement Learning
        self.rl_reward = torch.nn.Linear(embedding_size, 1)

        for p in self.parameters():
            p.requires_grad = True

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(config.DEVICE)

    def forward(self, image_batch, target_batch, attention_mask):
        N, seq_len = target_batch.size()
        # Image Encoder
        image_encoder_output_resnet = self.image_encoder_resnet(image_batch)  # Shape: (BATCH_SIZE, EMBEDDING_DIM)
        image_encoder_output_efficientnet = self.image_encoder_efficientnet(image_batch)  # Shape: (BATCH_SIZE, EMBEDDING_DIM)
        
        # Automatic selection of best encoder
        rl_reward_resnet = self.rl_reward(image_encoder_output_resnet)
        rl_reward_efficientnet = self.rl_reward(image_encoder_output_efficientnet)
        
        image_encoder_output = torch.where(rl_reward_resnet > rl_reward_efficientnet, image_encoder_output_resnet, image_encoder_output_efficientnet)
        image_encoder_output = image_encoder_output.unsqueeze(1)  # Shape: (BATCH_SIZE, 1, EMBEDDING_DIM)

        # Attention
        image_encoder_output, _ = self.attention(image_encoder_output, image_encoder_output, image_encoder_output)

        # Decoder
        decoder_output = self.decoder(input_ids=target_batch, 
                                      encoder_hidden_states=image_encoder_output, 
                                      attention_mask=attention_mask)['logits']

        return decoder_output
