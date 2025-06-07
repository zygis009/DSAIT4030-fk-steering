from typing import List, Optional
import jax.numpy as jnp
import flax.linen as nn
from transformers import BertConfig, BertTokenizer, ViTConfig, FlaxViTModel
from .bert import FlaxBertModel

class FlaxBLIP_Pretrain(nn.Module):
    image_size: int = 224
    vit: str = 'base'  # 'base' or 'large'
    med_config: str = 'med_config.json'  # mixture encoder-decoder conf
    embed_dim: int = 256
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 1. Configure and initialize Vision Transformer (FLaxViTModel)
        if self.vit == 'base':
            vision_width = 768
            depth = 12
            num_heads = 12
            drop_rate = 0
        elif self.vit == 'large':
            vision_width = 1024
            depth = 24
            num_heads = 16
            drop_rate = 0.1
        else:
            raise ValueError("vit parameter must be base or large")
        
        vit_config = ViTConfig(
            image_size=self.image_size,
            patch_size=16,
            hidden_size=vision_width,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            hidden_dropout_prob=drop_rate,
        )
        self.visual_encoder = FlaxViTModel(config=vit_config, dtype=self.dtype)

        # 2. Initialize and configure BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]

        # 3. Configure and initialize text encoder (FlaxBertModel) 
        bert_config = BertConfig.from_json_file(self.med_config)
        bert_config.encoder_width = vision_width  # cross attention _/\_, hopefully this shit works
        self.text_encoder = FlaxBertModel(config=bert_config, dtype=self.dtype)

        text_width = self.text_encoder.config.hidden_size
        
        # 3. BLIP-specific projection layers
        self.vision_proj = nn.Dense(vision_width, self.embed_dim, dtype=self.dtype)
        self.text_proj = nn.Dense(text_width, self.embed_dim, dtype=self.dtype)

    def __call__(
        self,
        images: jnp.ndarray,  # need (B, C, H, W)
        # Option 1: Pre-tokenized inputs (will be with CLIPTokenizer from FlaxStableDiffusionPipeline)
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        # Option 2: Raw string prompts (for use outside JIT or if FlaxBLIP_Module does tokenization)
        prompts: Optional[List[str]] = None,
        deterministic = True,
    ):
        
        if input_ids is None and prompts is not None:
            # text encode
            text_input = self.tokenizer(
                prompts,
                padding='max_length',
                truncation=True,
                max_length=35,
                return_tensors="np",
            )
            input_ids = text_input.input_ids
            attention_mask = text_input.attention_mask
        else:
            raise ValueError("No `prompts` or `(input_ids, attention_mask)` provided. Please provide at least one of these.")
        
        dropout_rng = None
        if not deterministic:
            dropout_rng = self.make_rng('dropout')

        image_embeds = self.visual_encoder(pixel_values=images, train=not deterministic, dropout_rng=dropout_rng).last_hidden_state
        
        # text encode cross attention with image
        image_atts = jnp.ones(image_embeds.shape[:-1], dtype=self.dtype)
        text_output = self.text_encoder(
            input_ids,  # Text input ids (B, seq_len)
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            train=not deterministic,
            dropout_rng=dropout_rng
        )

        return text_output.last_hidden_state[:, 0, :]  # (feature_dim)
