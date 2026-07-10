import torch
from torch import nn
from modules import ConvSC, Inception

from modules import ConvSC, Inception, ConvCfC          #added for cfc
from modules import ConvSC, Inception, ConvCfC, ConvCfCIncep    #added for cfc+inception
from modules import ConvSC, Inception, ConvCfC, ConvCfCIncep, CfCTemporalCell  #encoder & decoder

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, use_cfc=False, use_dfa=False):
        super(Encoder, self).__init__()
        self.use_cfc = use_cfc
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
        if use_cfc:
            self.cfc_cell = CfCTemporalCell(C_hid, use_dfa=use_dfa)

    def forward(self, x, T=None, ts=None):
        if not self.use_cfc:
            enc1 = self.enc[0](x)
            latent = enc1
            for i in range(1, len(self.enc)):
                latent = self.enc[i](latent)
            return latent, enc1

        else:
            B, T, C, H, W = x.shape
            skips = []
            latents = []
            h = None
            M = None

            for t in range(T):
                frame = x[:, t]

                enc1_t = self.enc[0](frame)
                lat_t = enc1_t
                for i in range(1, len(self.enc)):
                    lat_t = self.enc[i](lat_t)

                skips.append(enc1_t)

                if h is None:
                    h = torch.zeros_like(lat_t)

                if ts is None or t == 0:
                    ts_t = torch.ones(B, 1, 1, 1, device=x.device)
                else:
                    ts_t = ts[:, t - 1].view(B, 1, 1, 1)

                h, M = self.cfc_cell(lat_t, h, ts=ts_t, M=M)
                latents.append(h)

            latent = torch.stack(latents, dim=1)
            return latent, skips


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, use_cfc=False, use_dfa=False):
        super(Decoder, self).__init__()
        self.use_cfc = use_cfc
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        if use_cfc:
            self.cfc_cell = CfCTemporalCell(C_hid, use_dfa=use_dfa)

    def forward(self, hid, enc1=None, ts=None):
        if not self.use_cfc:
            for i in range(0, len(self.dec)-1):
                hid = self.dec[i](hid)
            Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
            Y = self.readout(Y)
            return Y

        else:
            B, T, C_hid, H_, W_ = hid.shape
            h = None
            M = None
            outputs = []

            for t in range(T):
                hid_t = hid[:, t]

                if h is None:
                    h = torch.zeros_like(hid_t)

                if ts is None or t == 0:
                    ts_t = torch.ones(B, 1, 1, 1, device=hid.device)
                else:
                    ts_t = ts[:, t - 1].view(B, 1, 1, 1)

                h, M = self.cfc_cell(hid_t, h, ts=ts_t, M=M)

                feat = h
                for i in range(0, len(self.dec)-1):
                    feat = self.dec[i](feat)

                skip_t = enc1[t]
                feat = self.dec[-1](torch.cat([feat, skip_t], dim=1))
                out_t = self.readout(feat)
                outputs.append(out_t)

            Y = torch.stack(outputs, dim=1).reshape(B*T, -1, outputs[0].shape[-2], outputs[0].shape[-1])
            return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8,
                 translator='inception', hid_S=64, bidirectional=False, use_dfa=False):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        self.translator = translator

        if translator == 'inception':
            enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
            for i in range(1, N_T-1):
                enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))

            dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
            for i in range(1, N_T-1):
                dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker=incep_ker, groups=groups))

            self.enc = nn.Sequential(*enc_layers)
            self.dec = nn.Sequential(*dec_layers)

        elif translator == 'cfc':
            self.cfc = ConvCfC(channel_in, channel_hid, N_T)

        elif translator == 'cfcincep':
            T = channel_in // hid_S
            self.cfc = ConvCfCIncep(channel_in, channel_hid, N_T, groups=groups, T=T,
                                     C_per_frame=hid_S, bidirectional=bidirectional, use_dfa=use_dfa)

    def forward(self, x, ts=None):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        if self.translator == 'inception':
            skips = []
            z = x
            for i in range(self.N_T):
                z = self.enc[i](z)
                if i < self.N_T - 1:
                    skips.append(z)

            z = self.dec[0](z)
            for i in range(1, self.N_T):
                z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        elif self.translator == 'cfc':
            z = self.cfc(x)

        elif self.translator == 'cfcincep':
            z = self.cfc(x, ts=ts)

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8,
                 incep_ker=[3,5,7,11], groups=8, translator='inception',
                 use_cfc_encdec=False, bidirectional=False, use_dfa=False):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.use_cfc_encdec = use_cfc_encdec
        self.enc = Encoder(C, hid_S, N_S, use_cfc=use_cfc_encdec, use_dfa=use_dfa)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups, translator=translator,
                             hid_S=hid_S, bidirectional=bidirectional, use_dfa=use_dfa)
        self.dec = Decoder(hid_S, C, N_S, use_cfc=use_cfc_encdec, use_dfa=use_dfa)

    def forward(self, x_raw, ts=None):
        B, T, C, H, W = x_raw.shape

        if not self.use_cfc_encdec:
            x = x_raw.view(B*T, C, H, W)
            embed, skip = self.enc(x)
            _, C_, H_, W_ = embed.shape
            z = embed.view(B, T, C_, H_, W_)
            hid = self.hid(z, ts=ts)
            hid = hid.reshape(B*T, C_, H_, W_)
            Y = self.dec(hid, skip)
            Y = Y.reshape(B, T, C, H, W)

        else:
            embed, skips = self.enc(x_raw, T=T, ts=ts)
            hid = self.hid(embed, ts=ts)
            Y = self.dec(hid, skips, ts=ts)
            Y = Y.reshape(B, T, C, H, W)

        return Y