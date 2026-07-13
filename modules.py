from torch import nn
import torch
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
    
#cfc implementation:
class ConvCfCCell(nn.Module):
    """
    Single ConvCfC cell — spatial adaptation of CfcCell from raminmh/CfC.
    Replaces all nn.Linear with nn.Conv2d(kernel=1) to preserve spatial dims.
    Backbone uses 3x3 convs (shared), four heads use 1x1 convs.
    """
    def __init__(self, channel_in, channel_hid):
        super(ConvCfCCell, self).__init__()

        # Backbone: takes [input, hidden] cat along channels
        # Mirrors: nn.Linear(input_size + hidden_size, backbone_units) + activation
        self.backbone = nn.Sequential(
            nn.Conv2d(channel_in + channel_hid, channel_hid, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel_hid),
            nn.SiLU(),
            nn.Conv2d(channel_hid, channel_hid, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel_hid),
            nn.SiLU(),
        )

        # Mirrors: self.ff1 — head g (candidate state)
        self.ff1 = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        # Mirrors: self.ff2 — head h (stable state)
        self.ff2 = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        # Mirrors: self.time_a and self.time_b — together form f (time constant)
        self.time_a = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        self.time_b = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, h, ts=1.0):
        """
        x_t: (B, channel_in, H, W)  — input at step t
        h:   (B, channel_hid, H, W) — hidden state
        ts:  scalar timestep (1.0 for uniform sampling like Moving MNIST)
        """
        combined = torch.cat([x_t, h], dim=1)   # (B, channel_in+channel_hid, H, W)
        feat = self.backbone(combined)            # (B, channel_hid, H, W)

        # Mirrors CfcCell forward exactly
        ff1 = self.tanh(self.ff1(feat))          # g: candidate state
        ff2 = self.tanh(self.ff2(feat))          # h: stable state
        t_a = self.time_a(feat)
        t_b = self.time_b(feat)
        t_interp = self.sigmoid(t_a * ts + t_b)  # f gate

        # CfC equation: x(t) = g*(1-gate) + h*gate
        new_h = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_h


class ConvCfC(nn.Module):
    """
    ConvCfC Translator.
    Receives (B, T*C, H, W) from SimVP encoder, processes recurrently over T steps,
    returns (B, T*C, H, W) to SimVP decoder. Mirrors Mid_Xnet's interface exactly.
    """
    def __init__(self, channel_in, channel_hid, N_T):
        super(ConvCfC, self).__init__()
        self.N_T = N_T

        # Project full T*C input down to channel_hid per step
        # We split T*C into N_T steps of C = channel_in // N_T channels each
        self.cell = ConvCfCCell(channel_in // N_T, channel_hid)

        # Project final hidden state back to channel_in for output
        self.output_proj = nn.Conv2d(channel_hid, channel_in // N_T, kernel_size=1)

    def forward(self, x):
        B, TC, H, W = x.shape
        C = TC // self.N_T

        x_steps = torch.chunk(x, self.N_T, dim=1)  # T tensors of (B, C, H, W)

        h = torch.zeros(B, self.cell.ff1.out_channels, H, W, device=x.device)

        # Collect all hidden states — mirrors return_sequences=True in Cfc.forward()
        hidden_states = []
        for t in range(self.N_T):
            h = self.cell(x_steps[t], h, ts=1.0)
            hidden_states.append(h)  # each (B, channel_hid, H, W)

        # Project each hidden state back to C channels and stack
        out = torch.stack([self.output_proj(h_t) for h_t in hidden_states], dim=1)  # (B, N_T, C, H, W)
        out = out.reshape(B, TC, H, W)
        return out
    

# #cfc + inception

class MultiScaleConv(nn.Module):
    """Inception-style multi-scale backbone for ConvCfC cell."""
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_3  = GroupConv2d(out_channels, out_channels, kernel_size=3,  padding=1,  stride=1, groups=groups, act_norm=True)
        self.branch_5  = GroupConv2d(out_channels, out_channels, kernel_size=5,  padding=2,  stride=1, groups=groups, act_norm=True)
        self.branch_7  = GroupConv2d(out_channels, out_channels, kernel_size=7,  padding=3,  stride=1, groups=groups, act_norm=True)
        self.branch_11 = GroupConv2d(out_channels, out_channels, kernel_size=11, padding=5,  stride=1, groups=groups, act_norm=True)

    def forward(self, x):
        x = self.pointwise(x)
        return self.branch_3(x) + self.branch_5(x) + self.branch_7(x) + self.branch_11(x)


    """
    CfC cell with Inception-style multi-scale backbone.
    use_dfa=True switches the gate from a per-step-only computation to the
    DFA-CfN dynamic feature accumulation form (Liang et al., NIALIM/DFA-CfN):
    the raw gating signal is accumulated across timesteps into M, and the
    sigmoid gate is computed from the accumulator instead of the current
    step alone.
    """


class ConvCfCIncepCell(nn.Module):
    """
    CfC cell with Inception-style multi-scale backbone.
    use_dfa=True switches the gate from a per-step-only computation to the
    DFA-CfN dynamic feature accumulation form. M is tracked as a running
    AVERAGE (not raw sum) to prevent unbounded growth from saturating the
    sigmoid gate over long recurrences.
    """
    def __init__(self, channel_in, channel_hid, groups=8, use_dfa=False):
        super(ConvCfCIncepCell, self).__init__()
        self.use_dfa = use_dfa

        self.backbone = nn.Sequential(
            MultiScaleConv(channel_in + channel_hid, channel_hid, groups=groups),
            MultiScaleConv(channel_hid, channel_hid, groups=groups),
        )

        self.ff1    = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)  # g
        self.ff2    = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)  # h
        self.time_a = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)  # decay term a(.)
        self.time_b = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)  # nonlinear term c(.)

        self.tanh    = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _omega_ts(ts):
        ts = torch.clamp(ts, min=1e-6)
        return torch.exp(ts * (1.0 - torch.log(ts)))

    def forward(self, x_t, h, ts=1.0, M=None, step_count=0):
        """
        M: running accumulator (unnormalized sum), or None at t=0
        step_count: how many steps have been accumulated into M so far
        Returns: new_h, new_M, new_step_count
        """
        combined = torch.cat([x_t, h], dim=1)
        feat = self.backbone(combined)

        ff1 = self.tanh(self.ff1(feat))
        ff2 = self.tanh(self.ff2(feat))
        t_a = self.time_a(feat)
        t_b = self.time_b(feat)

        if self.use_dfa:
            omega_ts = self._omega_ts(ts)
            Tinterp = t_a * omega_ts + t_b
            M = Tinterp if M is None else M + Tinterp
            step_count = step_count + 1
            gate = self.sigmoid(M / step_count)   # running AVERAGE, bounded magnitude
        else:
            gate = self.sigmoid(t_a * ts + t_b)
            M = None
            step_count = 0

        new_h = ff1 * (1.0 - gate) + gate * ff2
        return new_h, M, step_count


class ConvCfCIncep(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, groups=8, T=10,
                 C_per_frame=64, bidirectional=False, use_dfa=False):
        super(ConvCfCIncep, self).__init__()
        self.N_T = N_T
        self.T = T
        self.C_per_frame = C_per_frame
        self.bidirectional = bidirectional
        self.use_dfa = use_dfa

        self.cell = ConvCfCIncepCell(C_per_frame, channel_hid, groups=groups, use_dfa=use_dfa)

        if bidirectional:
            self.bwd_cell = ConvCfCIncepCell(C_per_frame, channel_hid, groups=groups, use_dfa=use_dfa)
            self.output_proj = nn.Conv2d(channel_hid * 2, C_per_frame, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(channel_hid, C_per_frame, kernel_size=1)

    def forward(self, x, ts=None):
        B, TC, H, W = x.shape
        x_steps = torch.chunk(x, self.T, dim=1)
        h = torch.zeros(B, self.cell.ff1.out_channels, H, W, device=x.device)
        M = None
        step_count = 0

        def _ts_at(t):
            if ts is None or t == 0:
                return torch.ones(B, 1, 1, 1, device=x.device)
            return ts[:, t - 1].view(B, 1, 1, 1)

        fwd_hidden = []
        for t in range(self.T):
            h, M, step_count = self.cell(x_steps[t], h, ts=_ts_at(t), M=M, step_count=step_count)
            fwd_hidden.append(h)

        if self.bidirectional:
            h_bwd = torch.zeros(B, self.cell.ff1.out_channels, H, W, device=x.device)
            M_bwd = None
            step_count_bwd = 0
            bwd_hidden = []
            for t in reversed(range(self.T)):
                if ts is None or t == self.T - 1:
                    ts_b = torch.ones(B, 1, 1, 1, device=x.device)
                else:
                    ts_b = ts[:, t].view(B, 1, 1, 1)
                h_bwd, M_bwd, step_count_bwd = self.bwd_cell(
                    x_steps[t], h_bwd, ts=ts_b, M=M_bwd, step_count=step_count_bwd)
                bwd_hidden.append(h_bwd)
            bwd_hidden = list(reversed(bwd_hidden))
            out = torch.stack([
                self.output_proj(torch.cat([fwd_hidden[t], bwd_hidden[t]], dim=1))
                for t in range(self.T)
            ], dim=1)
        else:
            out = torch.stack([self.output_proj(h_t) for h_t in fwd_hidden], dim=1)

        out = out.reshape(B, TC, H, W)
        return out
    

#for encoder & decoder
class CfCTemporalCell(nn.Module):
    def __init__(self, channel_hid, use_dfa=False):
        super(CfCTemporalCell, self).__init__()
        self.use_dfa = use_dfa
        self.backbone = nn.Sequential(
            nn.Conv2d(channel_hid * 2, channel_hid, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel_hid),
            nn.SiLU(),
            nn.Conv2d(channel_hid, channel_hid, kernel_size=3, padding=1),
            nn.GroupNorm(8, channel_hid),
            nn.SiLU(),
        )
        self.ff1    = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        self.ff2    = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        self.time_a = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        self.time_b = nn.Conv2d(channel_hid, channel_hid, kernel_size=1)
        self.tanh    = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _omega_ts(ts):
        ts = torch.clamp(ts, min=1e-6)
        return torch.exp(ts * (1.0 - torch.log(ts)))

    def forward(self, x_t, h, ts=1.0, M=None, step_count=0):
        combined = torch.cat([x_t, h], dim=1)
        feat = self.backbone(combined)
        ff1 = self.tanh(self.ff1(feat))
        ff2 = self.tanh(self.ff2(feat))
        t_a = self.time_a(feat)
        t_b = self.time_b(feat)

        if self.use_dfa:
            omega_ts = self._omega_ts(ts)
            Tinterp = t_a * omega_ts + t_b
            M = Tinterp if M is None else M + Tinterp
            step_count = step_count + 1
            gate = self.sigmoid(M / step_count)
        else:
            gate = self.sigmoid(t_a * ts + t_b)
            M = None
            step_count = 0

        new_h = ff1 * (1.0 - gate) + gate * ff2
        return new_h, M, step_count