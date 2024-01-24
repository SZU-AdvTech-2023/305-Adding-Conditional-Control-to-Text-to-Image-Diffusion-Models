import einops
from copy import deepcopy
import torch
import torch as th
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer, id_emb_ln
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor
from cldm.utils import frozen_module
from face_parsing.utlis import *

class ControlledUnetModel(UNetModel):
    def __init__(self,b1, b2, s1, s2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b1 = b1
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def Fourier_filter(self, x, threshold, scale):
        # FFT
        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W)).to(x.device)

        crow, ccol = H // 2, W // 2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        x_freq = x_freq * mask

        # IFFT
        x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            hs_ = hs.pop()
            if only_mid_control and control is not None:
                hs_ = torch.cat([h, hs.pop() + control.pop()], dim=1)

            # --------------- FreeU code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 1280:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                            hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:, :640] = h[:, :640] * ((self.b1 - 1) * hidden_mean + 1)
                hs_ = self.Fourier_filter(hs_, threshold=1, scale=self.s1)
            if h.shape[1] == 640:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                            hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:, :320] = h[:, :320] * ((self.b2 - 1) * hidden_mean + 1)
                hs_ = self.Fourier_filter(hs_, threshold=1, scale=self.s2)
            # ---------------------------------------------------------
            h = torch.cat([h, hs_], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            id_dim=None,  # arcface backbone support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        ## 增加线形层
                        if not use_spatial_transformer:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                        else:
                            layers.append(id_emb_ln(id_dim=id_dim, context_dim=context_dim))
                            layers.append(
                                SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint
                                ))
                        # layers.append(
                        #     AttentionBlock(
                        #         ch,
                        #         use_checkpoint=use_checkpoint,
                        #         num_heads=num_heads,
                        #         num_head_channels=dim_head,
                        #         use_new_attention_order=use_new_attention_order,
                        #     ) if not use_spatial_transformer else SpatialTransformer(
                        #         ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                        #         disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                        #         use_checkpoint=use_checkpoint
                        #     )
                        # )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ## 增加线性层
            id_emb_ln(id_dim=id_dim, context_dim=context_dim),

            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # self.input_hint_block把hint这一输入转化为320*64*64的尺寸
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    """
    this version add face parsing intricacy control based on v1.2.
    -- v1.3
    """

    def __init__(self, control_stage_config, id_emb_key, parsing_key, control_key, only_mid_control, processor_config,
                 parsing_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.id_emb_key = id_emb_key
        self.parsing_key = parsing_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # self.cond_encoder = nn.Sequential(OrderedDict([
        #     ("encoder", deepcopy(self.first_stage_model.encoder)),  # cond_encoder.encoder
        #     ("quant_conv", deepcopy(self.first_stage_model.quant_conv))  # cond_encoder.quant_conv
        # ]))
        # unfrozen_module(self.cond_encoder)
        # self.control_model.hint_encoder = self.cond_encoder
        # zero_module(self.cond_encoder)

        ## arcface extract id feature
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  ## id loss
        self.control_processor = instantiate_from_config(processor_config)
        frozen_module(self.control_processor)

        ## face parsing
        self.parsing_processor = instantiate_from_config(parsing_config)
        frozen_module(self.parsing_processor)
        # self.SSIM = SSIM()
        # self.parsing_loss = OhemCELoss(parsing_config.loss_thresh, parsing_config.loss_n_min,
        #                                parsing_config.loss_ignore_index).to(self.device)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        ## id_feature
        id_emb = batch[self.id_emb_key]
        if bs is not None:
            id_emb = id_emb[:bs]
        id_emb = id_emb.to(self.device)
        # with torch.no_grad():
            # self.control_processor.eval()
        id_feature, total_feature = self.control_processor(id_emb)

        ## parsing ground_truth
        parsing_gt = batch[self.parsing_key]
        if bs is not None:
            parsing_gt = parsing_gt[:bs]
        parsing_gt = parsing_gt.to(self.device)
        # parsing_gt = torch.squeeze(parsing_gt, 1)
        parsing_gt = parsing_gt.to(memory_format=torch.contiguous_format)

        return x, dict(c_crossattn=[c], c_concat=[control], id_emb=[total_feature], parsing_gt=[parsing_gt])

    # ControlledUnetModel继承了LatentDiffusion，此处重写apply_model方法，p_loss计算时模型调用就会调用新的apply_model前向计算output
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_id = torch.cat(cond['id_emb'], 1)
        # cond_id = cond_id.reshape(-1,1,cond_id.shape[-1])

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_id)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N] #初始
        # 增加 id
        c_cat, c, c_id = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["id_emb"][0][:N]
        # ----
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            # uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross] 初始
            # 增加id_emb
            uc_id = c_id
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "id_emb": [uc_id]}
            # ----
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "id_emb": [c_id]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8) ##
        # shape = (self.channels, h, w)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())

        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            DDIMSampler(self)
        # ddim_sampler = DDIMSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        # shape = (self.channels, h // 8, w // 8)
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        # return samples, intermediates

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # 增加一步采样
        one_to_end = (x_noisy - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                                    x_start.shape) * model_output) / extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape)
        one_to_end = self.decode_first_stage(one_to_end)
        # [0,1]
        one_to_end_max = torch.max(one_to_end.reshape(one_to_end.shape[0], one_to_end.shape[1], -1),dim = -1).values.unsqueeze(-1).unsqueeze(-1)
        one_to_end_min = torch.min(one_to_end.reshape(one_to_end.shape[0], one_to_end.shape[1], -1),dim = -1).values.unsqueeze(-1).unsqueeze(-1)
        one_to_end = (one_to_end-one_to_end_min)/(one_to_end_max-one_to_end_min)

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        T = torch.tensor(self.num_timesteps, device=self.device).expand((t.shape))
        adjust_w = torch.tensor([(1 if t[i] <= 500 else 0) for i in range(t.shape[0])]).to(self.device)

        # 细腻度损失
        # parsing, _, _ = self.parsing_processor(parsing_trans(one_to_end))
        # loss_parsing = self.get_parsing_loss(parsing, cond["parsing_gt"][0], one_to_end, x_start) * adjust_w
        # loss_parsing = loss_parsing.mean()
        # loss_dict.update({f'{prefix}/loss_id': loss_parsing})
        # loss += 0.01 * loss_parsing
        # loss_dict.update({f'{prefix}/loss': loss})

        # id loss
        id_loss = self.id_loss(one_to_end, cond['id_emb'][0][:,0])
        id_loss = id_loss.to(self.device) * (1.0 - (t / T)) * adjust_w
        id_loss = id_loss.mean()
        loss_dict.update({f'{prefix}/loss_id': id_loss})
        loss += 5 * id_loss
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def id_loss(self, x0_hat, gt_id):
        # arcface transform
        x0_hat = F.interpolate(x0_hat, size=(112, 112), mode='bilinear', align_corners=False)
        x0_hat_id, _ = self.control_processor(self.normalize(x0_hat))

        loss = 1 - torch.cosine_similarity(x0_hat_id, gt_id, dim=1)

        return loss

    def get_parsing_loss(self, parsing, parsing_gt, x0_hat, x0):
        """

        Args:
            parsing: argmin后的parsing
            parsing_gt:
            x0_hat: one_to_end
            x0: x_start

        Returns:
            SSIM_loss:
        """
        loss = 0
        x0 = self.decode_first_stage(x0)
        x0 = (x0 - x0.min()) / (x0.max() - x0.min())
        parsing = parsing.argmax(1)
        num_of_class = torch.max(parsing)
        for pi in range(1, int(num_of_class) + 1):
            index = torch.where(parsing==pi)[-2:]
            mask = (parsing == pi)
            mask_img= x0_hat * mask.unsqueeze(1)

            index2 = torch.where(parsing_gt==pi)
            mask = (parsing_gt == pi)
            mask_img2 = x0 * mask.unsqueeze(1)
            if len(index[0])>0 and len(index2[0])>0:
                box, box2 = bounding_box(mask_img, mask_img2, index, index2)
                loss += self.SSIM(box,box2)

        return loss