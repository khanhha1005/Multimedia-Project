import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config

from dataset import Sift1mDataset


class EnhancedProjection(nn.Module):
    """Enhanced projection with residual connections, LayerNorm, and GELU activation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        # First layer
        h = self.input_proj(x)
        h = self.layer_norm1(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second layer with residual
        h2 = self.hidden_proj(h)
        h2 = self.layer_norm2(h2)
        h2 = self.activation(h2)
        h2 = self.dropout(h2)
        h = h + h2  # Residual connection
        
        # Output layer
        out = self.output_proj(h)
        out = self.layer_norm3(out)
        
        # Global residual
        residual = self.residual_proj(x)
        out = out + residual
        
        return out


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pos_embedding(positions)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss for better generalization."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Create smooth labels
        smooth_labels = torch.full_like(logprobs, self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = (-smooth_labels * logprobs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class T5ForPretrainV2(T5ForConditionalGeneration):
    """
    Enhanced T5 model for neural retrieval with:
    - Enhanced projection layers with residual connections
    - Label smoothing for better generalization
    - Position-aware decoding
    - Temperature scaling for calibrated predictions
    - Optional retrieval loss support
    """
    
    def __init__(self, config: T5Config, args):
        super(T5ForPretrainV2, self).__init__(config)
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.args = args
        self.config: T5Config
        
        # Get dimensions
        input_dim = Sift1mDataset.VECTOR_DIM // args.num_subspace
        hidden_dim = self.config.d_model
        output_dim = self.config.d_model
        
        # Enhanced projection network
        dropout = getattr(args, 'proj_dropout', args.dropout_rate)
        self.output_proj = EnhancedProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Positional encoding for hierarchical PQ structure
        self.pos_encoding = PositionalEncoding(self.config.d_model, max_len=args.num_subspace + 1)
        
        # Temperature for logits scaling
        self.temperature = nn.Parameter(torch.ones(1) * getattr(args, 'temperature', 1.0))
        
        # Loss function selection
        label_smoothing = getattr(args, 'label_smoothing', 0.1)
        loss_type = getattr(args, 'loss_type', 'label_smoothing')
        
        if loss_type == 'label_smoothing':
            self.loss_fct = LabelSmoothingLoss(args.num_clusters, smoothing=label_smoothing)
        elif loss_type == 'focal':
            self.loss_fct = FocalLoss(gamma=2.0, alpha=0.25)
        else:
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optional: retrieval loss weight
        self.retrieval_loss_weight = getattr(args, 'retrieval_loss_weight', 0.0)
        
        # Subspace-specific heads for better specialization
        use_subspace_heads = getattr(args, 'use_subspace_heads', False)
        if use_subspace_heads:
            self.subspace_heads = nn.ModuleList([
                nn.Linear(self.config.d_model, args.num_clusters) for _ in range(args.num_subspace)
            ])
        else:
            self.subspace_heads = None

    def forward(
        self,
        decoder_inputs_embeds,
        encoder_outputs=None,
        inputs_embeds=None,
        labels=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        subspace_idx=None,  # For subspace-specific prediction
    ):
        '''
            Args
                decoder_inputs_embeds:  vecids
                subspace_idx: current subspace index for specialized heads
            Return
                Seq2SeqLMOutput()
        '''

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # Apply positional encoding
        decoder_inputs_embeds = self.pos_encoding(decoder_inputs_embeds)

        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,            
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # Use subspace-specific head if available
        if self.subspace_heads is not None and subspace_idx is not None:
            lm_logits = self.subspace_heads[subspace_idx](sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)
        
        # Apply temperature scaling
        lm_logits = lm_logits / self.temperature

        loss = None
        if labels is not None:
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            return loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


# Alias for backwards compatibility with train.py
T5ForPretrain = T5ForPretrainV2

