import torch
from torch import nn

from ..cnn.eegnet import EEGNet


class _EEGNetAttentionBackbone(EEGNet):
    def __init__(self, chunk_size: int, num_electrodes: int, F1: int, F2: int,
                 D: int, kernel_1: int, kernel_2: int, dropout: float,
                 heads: int, attn_dropout: float, use_attention: bool):
        super(_EEGNetAttentionBackbone, self).__init__(
            chunk_size=chunk_size,
            num_electrodes=num_electrodes,
            F1=F1,
            F2=F2,
            D=D,
            num_classes=2,
            kernel_1=kernel_1,
            kernel_2=kernel_2,
            dropout=dropout)
        del self.lin

        with torch.no_grad():
            mock = self.block1(torch.zeros(1, 1, num_electrodes, chunk_size))
            mock = self.block2(mock)

        self.use_attention = use_attention
        self.num_steps = mock.shape[3]
        if use_attention:
            self.embed_dim = mock.shape[1]
            if self.embed_dim % heads != 0:
                raise ValueError(
                    f'EEGNet channels {self.embed_dim} must be divisible by '
                    f'the number of heads {heads}.')
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.num_steps, self.embed_dim))
            self.in_norm = nn.LayerNorm(self.embed_dim)
            self.attention = nn.MultiheadAttention(self.embed_dim,
                                                   heads,
                                                   dropout=attn_dropout,
                                                   batch_first=True)
            self.out_norm = nn.LayerNorm(self.embed_dim)
            self.dropout = nn.Dropout(attn_dropout)
        else:
            self.embed_dim = mock.shape[1] * self.num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        if not self.use_attention:
            return x.flatten(start_dim=1)
        x = x.squeeze(2).transpose(1, 2)
        x = self.in_norm(x + self.pos_embedding)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.out_norm(x + self.dropout(attended))
        return x.mean(dim=1)


class EEGNetAttentionDANN(nn.Module):
    r'''
    An EEGNet backbone with temporal multi-head attention and a domain-adversarial head.

    The backbone, label classifier and domain classifier are exposed separately so
    that the model can be passed directly to :class:`torcheeg.trainers.DANNTrainer`,
    which applies the gradient reversal layer during training.

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.models import EEGNetAttentionDANN
        from torcheeg.trainers import DANNTrainer

        model = EEGNetAttentionDANN(chunk_size=128,
                                    num_electrodes=32,
                                    num_classes=2)

        trainer = DANNTrainer(model.backbone,
                              model.classifier,
                              model.domain_classifier,
                              num_classes=2,
                              devices=1,
                              accelerator='cpu')

    Args:
        chunk_size (int): Number of data points included in each EEG chunk. (default: :obj:`128`)
        num_electrodes (int): The number of electrodes in the EEG signal. (default: :obj:`32`)
        F1 (int): The filter number of block 1 in EEGNet. (default: :obj:`8`)
        F2 (int): The filter number of block 2 in EEGNet. (default: :obj:`16`)
        D (int): The depth multiplier of EEGNet. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1 in EEGNet. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2 in EEGNet. (default: :obj:`16`)
        dropout (float): Probability of an element to be zeroed in the EEGNet dropout layers. (default: :obj:`0.25`)
        heads (int): The number of attention heads. (default: :obj:`4`)
        attn_dropout (float): Probability of an element to be zeroed in attention and embedding dropout. (default: :obj:`0.1`)
        domain_hidden_channels (int): The number of hidden nodes in the domain classifier. (default: :obj:`64`)
        use_attention (bool): Whether to apply temporal multi-head attention after the EEGNet blocks. (default: :obj:`True`)
        use_dann (bool): Whether to create the domain classifier head for adversarial training. (default: :obj:`True`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
    '''

    def __init__(self,
                 chunk_size: int = 128,
                 num_electrodes: int = 32,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 heads: int = 4,
                 attn_dropout: float = 0.1,
                 domain_hidden_channels: int = 64,
                 use_attention: bool = True,
                 use_dann: bool = True,
                 num_classes: int = 2):
        super(EEGNetAttentionDANN, self).__init__()
        self.backbone = _EEGNetAttentionBackbone(chunk_size=chunk_size,
                                                 num_electrodes=num_electrodes,
                                                 F1=F1,
                                                 F2=F2,
                                                 D=D,
                                                 kernel_1=kernel_1,
                                                 kernel_2=kernel_2,
                                                 dropout=dropout,
                                                 heads=heads,
                                                 attn_dropout=attn_dropout,
                                                 use_attention=use_attention)
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)
        if use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.backbone.embed_dim, domain_hidden_channels),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(domain_hidden_channels, 2))
        else:
            self.domain_classifier = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal with shape :obj:`[batch_size, 1, num_electrodes, chunk_size]`.

        Returns:
            torch.Tensor: Feature representation with shape :obj:`[batch_size, embed_dim]`.
        '''
        return self.backbone(x)

    def predict_domain(self, features: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            features (torch.Tensor): Features returned by :meth:`extract_features`.

        Returns:
            torch.Tensor: Domain logits with shape :obj:`[batch_size, 2]`.
        '''
        if self.domain_classifier is None:
            raise ValueError('The model was created with use_dann=False.')
        return self.domain_classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal with shape :obj:`[batch_size, 1, num_electrodes, chunk_size]`.

        Returns:
            torch.Tensor: Classification logits with shape :obj:`[batch_size, num_classes]`.
        '''
        return self.classifier(self.extract_features(x))
