#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F                          # [NEW] 用于 mask 插值对齐

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from .srm_encoder import SRMVisionTower, SRMProjector


# ==============================================================
# [NEW] 掩码生成头：把 SRM 的空间特征图转成引导 CLIP 的 patch mask
# 单独定义成一个小模块，方便在 initialize_vision_modules 里初始化
# ==============================================================
class SRMMaskHead(nn.Module):
    """
    输入：SRM 的空间特征图 (B, srm_hidden_size, H_srm, W_srm)
    输出：与 CLIP patch token 对齐的软掩码 (B, N_clip_patches)

    只有一个 1×1 Conv + Sigmoid，参数量极少，不会给训练增加负担。
    """
    def __init__(self, srm_hidden_size: int, num_clip_patches: int = 576):
        super().__init__()
        # 1×1 Conv 把通道数压到 1，再 Sigmoid 得到 [0,1] 软掩码
        self.conv = nn.Conv2d(srm_hidden_size, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.num_clip_patches = num_clip_patches  # 24×24=576（ViT-L/14@336）

    def forward(self, srm_spatial_feat: torch.Tensor) -> torch.Tensor:
        """
        srm_spatial_feat: (B, C, H, W)  SRM CNN 中间层的空间特征图
        返回: (B, N)  N = num_clip_patches，软掩码，值域 [0,1]
        """
        # 1×1 Conv → (B, 1, H, W)
        mask = self.sigmoid(self.conv(srm_spatial_feat))  # (B, 1, H, W)

        # 插值到 CLIP patch 分辨率（24×24）
        n = int(self.num_clip_patches ** 0.5)             # 24
        mask = F.interpolate(
            mask, size=(n, n), mode='bilinear', align_corners=False
        )                                                  # (B, 1, 24, 24)

        # 展平成序列，对应 CLIP 的 patch token 顺序
        mask = mask.squeeze(1).flatten(1)                  # (B, 576)
        return mask


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # ==========================================================
        # 【毕设新增代码】：初始化 SRM 分支、Projector2、掩码头、引导强度
        # ==========================================================
        if getattr(self, 'srm_tower', None) is None:
            self.srm_tower = SRMVisionTower()
            self.srm_projector = SRMProjector(
                srm_hidden_size=self.srm_tower.hidden_size,
                llm_hidden_size=self.config.hidden_size
            )

            # [NEW] 掩码生成头
            # srm_tower.hidden_size 是 SRM CNN 最后一层的通道数
            # num_clip_patches=576 对应 ViT-L/14@336（24×24 个 patch）
            self.srm_mask_head = SRMMaskHead(
                srm_hidden_size=self.srm_tower.hidden_size,
                num_clip_patches=576,
            )

            # [NEW] 可学习的引导强度 α，初始值 0.5
            # 让模型自己学习"SRM 对 CLIP 的影响应该有多强"
            self.mask_guidance_alpha = nn.Parameter(
                torch.tensor(0.5)
            )
        else:
            # 确保在 LoRA 冻结时，新模块依然保持可训练
            for p in self.srm_projector.parameters():
                p.requires_grad = True
            for p in self.srm_mask_head.parameters():       # [NEW]
                p.requires_grad = True
            self.mask_guidance_alpha.requires_grad = True   # [NEW]
        # ==========================================================

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # ==========================================================
    # 【毕设修改】encode_images：加入掩码引导逻辑
    #
    # 完整数据流：
    #   images
    #     ├─► CLIP-ViT ──────────────────────────────────────────────────────┐
    #     │                                                                   │
    #     └─► SRM Filter → CNN Encoder                                        │
    #                           ├─► srm_mask_head → spatial_mask (B,576)     │
    #                           │        ↓  点乘引导 (⊗)                      │
    #                           │   CLIP patch tokens × (1 + α×mask) ────────┘
    #                           │        ↓
    #                           │   mm_projector → semantic_tokens (B,N,llm_dim)
    #                           │
    #                           └─► srm_projector → freq_tokens (B,M,llm_dim)
    #
    #   fused = cat(semantic_tokens, freq_tokens)  →  LLM
    # ==========================================================
    def encode_images(self, images, srm_images=None):
        model = self.get_model()

        # ── Step 1: CLIP-ViT 提取 patch tokens ─────────────────
        # clip_tokens shape: (B, N+1, clip_hidden)
        # N+1 = 1(CLS) + 576(patches)，ViT-L/14@336
        clip_tokens = model.get_vision_tower()(images)   # (B, 577, 1024)

        # ── Step 2: SRM 分支 ────────────────────────────────────
        srm_tower     = getattr(model, 'srm_tower',     None)
        srm_projector = getattr(model, 'srm_projector', None)
        srm_mask_head = getattr(model, 'srm_mask_head', None)
        alpha         = getattr(model, 'mask_guidance_alpha', None)

        if srm_tower is not None and srm_projector is not None:
            actual_srm_inputs = srm_images if srm_images is not None else images

            # 对齐设备与精度（半精度训练时必须）
            srm_tower.to(device=actual_srm_inputs.device,
                         dtype=actual_srm_inputs.dtype)
            srm_projector.to(device=actual_srm_inputs.device,
                             dtype=actual_srm_inputs.dtype)

            # SRM 前向：返回 (srm_feat, srm_spatial)
            # srm_feat:    (B, M, srm_hidden)  用于 Projector2
            # srm_spatial: (B, srm_hidden, H_srm, W_srm)  用于生成 mask
            # ──────────────────────────────────────────────────
            # 注意：你需要让 SRMVisionTower.forward() 同时返回这两个值
            # 如果原来只返回 srm_feat，见文件末尾的修改说明
            # ──────────────────────────────────────────────────
            srm_feat, srm_spatial = srm_tower(actual_srm_inputs)

            # ── Step 3: 掩码生成 ──────────────────────────────
            if srm_mask_head is not None and alpha is not None:
                srm_mask_head.to(device=actual_srm_inputs.device,
                                 dtype=actual_srm_inputs.dtype)

                # spatial_mask: (B, 576)，值域 [0,1]
                # 1 = SRM 认为该 patch 频率异常（疑似伪造）
                # 0 = 该 patch 频率正常
                spatial_mask = srm_mask_head(srm_spatial)  # (B, 576)

                # ── Step 4: 掩码引导 CLIP patch tokens ──────────
                # clip_tokens[:, 0, :]  → CLS token，不动
                # clip_tokens[:, 1:, :] → patch tokens，用 mask 增强
                #
                # 公式：patch' = patch × (1 + α × mask)
                #   mask=0 → patch' = patch        （正常区域，原样保留）
                #   mask=1 → patch' = patch × (1+α) （异常区域，额外强调）
                cls_token    = clip_tokens[:, :1, :]   # (B, 1, 1024)
                patch_tokens = clip_tokens[:, 1:, :]   # (B, 576, 1024)

                # α 限幅防止梯度爆炸
                alpha_clamped = torch.clamp(alpha, 0.0, 2.0)

                # mask: (B, 576) → (B, 576, 1) 广播到特征维度
                mask_expanded = spatial_mask.unsqueeze(-1)          # (B, 576, 1)
                guided_patches = patch_tokens * (1.0 + alpha_clamped * mask_expanded)

                # 拼回 CLS token，形状与原始 CLIP 输出完全一致
                clip_tokens = torch.cat([cls_token, guided_patches], dim=1)
                # (B, 577, 1024)，可以直接送进原有 mm_projector

            # ── Step 5: Projector1 —— 引导后的语义 tokens ────────
            # mm_projector 把 (B, 577, 1024) → (B, 577, llm_hidden)
            image_features = model.mm_projector(clip_tokens)   # (B, 577, llm_dim)

            # ── Step 6: Projector2 —— 频率物证 tokens ────────────
            srm_features = srm_projector(srm_feat)             # (B, M, llm_dim)

            # ── Step 7: 融合（token 级别 concat）─────────────────
            # 最终 token 序列 = [语义 tokens | 频率 tokens]
            # LLM 既能看到"哪里看起来不自然"又能看到"具体的频率物证"
            image_features = torch.cat([image_features, srm_features], dim=1)
            # (B, 577+M, llm_dim)

        else:
            # 退化模式：没有 SRM 分支时走原有逻辑（兼容单模态推理）
            image_features = model.mm_projector(clip_tokens)

        return image_features

    # ==========================================================
    # 【毕设修改】：增加 srm_images 参数，透传给 encode_images
    # 其余逻辑与原版完全相同，只在两处调用 encode_images 的地方加了参数
    # ==========================================================
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, srm_images=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            # [MOD] 透传 srm_images
            image_features = self.encode_images(concat_images, srm_images=srm_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # [MOD] 透传 srm_images
            image_features = self.encode_images(images, srm_images=srm_images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


# ==============================================================
# [NEW] SRMVisionTower 修改说明
# ==============================================================
# 你的 srm_encoder.py 里的 SRMVisionTower.forward() 目前只返回
# srm_feat（用于 Projector2）。
#
# 为了让 encode_images 能拿到 srm_spatial（用于生成掩码），
# 需要让 forward() 同时返回两个值。
#
# 在你的 srm_encoder.py 里找到 SRMVisionTower.forward()，改成：
#
#   def forward(self, x):
#       srm_residual = self.srm_filter(x)      # SRM 固定滤波
#       srm_spatial  = self.cnn_encoder(srm_residual)  # CNN 空间特征图
#       srm_feat     = self.pool(srm_spatial)  # 全局池化 → token 序列
#       return srm_feat, srm_spatial           # ← 返回两个值
#
# 如果你的 CNN encoder 和 pool 是一整个 nn.Sequential，
# 把它拆成两段，中间截出 srm_spatial 即可。
# ==============================================================
