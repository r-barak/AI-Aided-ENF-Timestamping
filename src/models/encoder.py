import torch
import torch.nn as nn


class StatsPool1D(nn.Module):
    """
    Computes the mean and standard deviation across the temporal dimension,
    concatenating them to form a fixed-size representation.
    """
    def forward(self, x):
        m = x.mean(dim=-1)
        s = x.std(dim=-1, unbiased=False)
        return torch.cat([m, s], dim=1)

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, dilation=1, groups=8):
        super().__init__()
        pad = (k // 2) * dilation

        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.act   = nn.GELU()

        # 1x1 conv to match dimensions for the residual connection if needed
        self.proj  = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.act(self.gn2(self.conv2(y)))
        return self.act(y + self.proj(x))

class TCNStack(nn.Module):
    """
    A stack of TCN blocks with exponentially increasing dilations to expand the receptive field.
    """
    def __init__(self, c_in, c_out, k=5, dilations=(1,2,4,8), groups=8):
        super().__init__()
        blocks = []
        c = c_in
        for d in dilations:
            blocks.append(TCNBlock(c, c_out, k=k, dilation=d, groups=groups))
            c = c_out
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    """
    A non-linear projection head used to map features into the contrastive embedding space.
    """
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        return self.fc2(h)

class TwoStageTCNEncoder(nn.Module):
    """
    The two-stage TCN architecture.
    Stage 1: Captures local/mid-range patterns.
    The down-sampling layer reduces sequence length and increases channels.
    Stage 2: Captures long-range dependencies on the compressed sequence.
    Finally, the projection head maps the global statistics to the embedding space.
    """
    def __init__(
        self,
        in_channels=3,
        c1=64,                # channels after stem
        c2=128,               # channels for stage2
        k=5,
        dilations1=(1,2,4,8),
        dilations2=(1,2,4,8),
        groups=8,
        embedding_dim=128,
        proj_dim=128
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=min(groups, c1), num_channels=c1),
            nn.GELU())

        self.stage1 = TCNStack(c1, c1, k=k, dilations=dilations1, groups=groups)

        self.down = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), # T -> T/2
            nn.Conv1d(c1, c2, kernel_size=1),
            nn.GroupNorm(num_groups=min(groups, c2), num_channels=c2),
            nn.GELU())

        self.stage2 = TCNStack(c2, c2, k=k, dilations=dilations2, groups=groups)

        self.pool = StatsPool1D()
        self.fc = nn.Linear(2*c2, embedding_dim) # 2*c2 because StatsPool1D concatenates mean and std
        self.head = ProjectionHead(embedding_dim, hidden_dim=2*embedding_dim, out_dim=proj_dim)
        self.bn = nn.BatchNorm1d(num_features=proj_dim, affine=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        pooled = self.pool(x)
        h = self.fc(pooled)
        h = self.head(h)
        h = self.bn(h)
        return h