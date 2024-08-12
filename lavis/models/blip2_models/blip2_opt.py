"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer

import copy
from einops import rearrange, repeat
from torch.nn.parameter import Parameter
import random


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        freeze_qformer=False,
        instruct_qformer=True,
        task_name='',
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.instruct_qformer = instruct_qformer
        self.task_name = task_name
        if not instruct_qformer:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")


        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>"]}
        )
        self.endofchunk_token_id = self.opt_tokenizer("<|endofchunk|>", add_special_tokens=False)[
            "input_ids"
        ][-1]
        opt_config = OPTConfig.from_pretrained(opt_model)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, config=opt_config, torch_dtype=torch.float16 
        )
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))
            
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.float()
        logging.info("freeze opt")
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def split_caption(self, caption, split_ratio=0.5):
        if '|' in caption:
            caption_pre, caption_suf = caption.split('|')
            caption_pre, caption_suf = caption_pre, caption_suf.strip()
        elif isinstance(self.prompt, list):
            prompt = random.choice(self.prompt)
            caption_pre = prompt
            caption_suf = caption
        else:
            caption_pre = self.prompt
            caption_suf = caption
        return caption_pre, caption_suf

    def forward(self, samples):

        # Caption
        self.prompt = list(self.prompt)
        
        group_index = [] 
        group_index_i = []
        group_index_ii = [0]
        text_index = []
        text_index_i = []
        if samples['text_input'][0] != 'none':
            text_index_i.append(0)
            group_index_i.append(group_index_ii)
            group_index_ii = []
        for dialog_id_i in range(1, len(samples['dialog_id'])):
            if samples['dialog_id'][dialog_id_i] == samples['dialog_id'][dialog_id_i-1]:
                group_index_ii.append(dialog_id_i)
                if samples['text_input'][dialog_id_i] != 'none':
                    text_index_i.append(dialog_id_i)
                    group_index_i.append(group_index_ii)
                    group_index_ii = []
            else:
                group_index.append(group_index_i)
                text_index.append(text_index_i)
                if samples['dialog_id'][dialog_id_i] == -1:
                    group_index_i = []
                    break
                group_index_ii = [dialog_id_i]
                group_index_i = []
                text_index_i = []
                if samples['text_input'][dialog_id_i] != 'none':
                    text_index_i.append(dialog_id_i)
                    group_index_i.append(group_index_ii)
                    group_index_ii = []
        if group_index_i:
            group_index.append(group_index_i)
            text_index.append(text_index_i)
        group_num = len(group_index)

        samples['text_output'] = []
        for gi in range(group_num):
            caption = samples['text_input'][group_index[gi][-1][-1]]
            caption_pre, caption_suf = self.split_caption(caption)
            samples['text_input'][group_index[gi][-1][-1]] = caption_pre
            samples['text_output'].append(caption_suf)
        
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        
        
        img_embeds_list = [] 
        text_input_list = []
        pred_idxs = []
        for i in range(group_num):
            for j in range(len(group_index[i])):
                img_embeds_i = []
                for k in group_index[i][j]:
                    img_embeds_i.append(image_embeds[k])
                    if samples['text_input'][k] != 'none':
                        text_input_list.append(samples['text_input'][k])
                img_embeds_list.append(torch.stack(img_embeds_i))
            pred_idxs.append(len(text_input_list) - 1)
        samples['text_input'] = text_input_list 
                    
        ### VQA
        # if self.prompt:
        #     for i in range(len(samples['text_input'])):
        #         if '|' in samples['text_input'][i]:
        #             q, a = samples['text_input'][i].split('|')
        #             samples['text_input'][i] = self.prompt.format(q) + a
        #         else:
        #             samples['text_input'][i] = self.prompt.format(samples['text_input'][i])

        if self.instruct_qformer:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

        query_output_list = []            
        for i in range(len(img_embeds_list)):
            img_embeds_i = img_embeds_list[i]

            img_embeds_i = rearrange(
                img_embeds_i, "b v d -> 1 (b v) d"
            )

            image_atts_i = torch.ones(img_embeds_i.size()[:-1], dtype=torch.long).to(
                image.device 
            )   
            query_tokens = self.query_tokens  

            if self.instruct_qformer:
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask[i].unsqueeze(0)], dim=1)
                
                query_output_i = self.Qformer.bert(
                    text_Qformer.input_ids[i].unsqueeze(0),
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=img_embeds_i,
                    encoder_attention_mask=image_atts_i,
                    return_dict=True,
                )
                query_output_i = query_output_i.last_hidden_state
                query_output_list.append(query_output_i)
            else:
                query_output_i = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=img_embeds_i,
                    encoder_attention_mask=image_atts_i,
                    return_dict=True,
                )  
                query_output_list.append(query_output_i.last_hidden_state)
        
        
        query_output = torch.cat(query_output_list, dim=0)
        
        inputs_opt = self.opt_proj(query_output) 
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"
        
        with self.maybe_autocast(dtype=torch.float):
            input_tokens = self.opt_tokenizer(
                samples["text_input"],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            text = [t + "\n" for t in samples["text_output"]]
            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            opt_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            all_inputs_embeds = self.opt_model.model.decoder.embed_tokens(input_tokens.input_ids) # torch.Size([32, 20, 2048])
            eoc_embed = self.opt_model.model.decoder.embed_tokens(torch.tensor(self.endofchunk_token_id).to(image.device)).unsqueeze(0)
            eoc_att = torch.ones(1, dtype=torch.long).to(image.device)


            inputs_embeds_list = []
            encoder_atts_list = []
            start_index = 0
            targets_list = []
            
            for gi in range(group_num):
                inputs_embeds_i = []
                encoder_atts_i = []
                for gj in range(len(group_index[gi])):
                    
                    if self.instruct_qformer:
                        if gj == len(group_index[gi]) - 1:
                            inputs_embeds_i.append(inputs_opt[start_index+gj, :self.query_tokens.shape[1]])
                            encoder_atts_i.append(atts_opt[start_index+gj, :self.query_tokens.shape[1]]) 
                            inputs_embeds_i.append(all_inputs_embeds[start_index+gj])
                            encoder_atts_i.append(input_tokens.attention_mask[start_index+gj])
                        else:
                            inputs_embeds_i.append(inputs_opt[start_index+gj, self.query_tokens.shape[1]:])
                            encoder_atts_i.append(atts_opt[start_index+gj, self.query_tokens.shape[1]:]) 
                    else:
                        inputs_embeds_i.append(inputs_opt[start_index+gj])
                        encoder_atts_i.append(atts_opt[start_index+gj])
                        inputs_embeds_i.append(all_inputs_embeds[start_index+gj])
                        encoder_atts_i.append(input_tokens.attention_mask[start_index+gj])

                    if gj != len(group_index[gi]) - 1:
                        inputs_embeds_i.append(copy.deepcopy(eoc_embed))
                        encoder_atts_i.append(copy.deepcopy(eoc_att))
                inputs_embeds_list.append(torch.cat(inputs_embeds_i, dim=0)) 
                encoder_atts_i = torch.cat(encoder_atts_i, dim=0)
                encoder_atts_list.append(encoder_atts_i)
                targets_list.append(torch.ones(encoder_atts_i.size(), dtype=torch.long).to(image.device).fill_(-100))
                start_index += len(group_index[gi])


            tmp = []
            for i in range(len(targets)):
                inputs_embeds_list[i] = torch.cat((inputs_embeds_list[i], opt_embeds[i]), dim=0).unsqueeze(0)
                encoder_atts_list[i] = torch.cat((encoder_atts_list[i], opt_tokens.attention_mask[i]), dim=0).unsqueeze(0)
                tmp.append(torch.cat((targets_list[i], targets[i]), dim=0).unsqueeze(0))
            targets = tmp

            

            try:
                inputs_embeds = torch.cat(inputs_embeds_list, dim=0) 
                encoder_atts = torch.cat(encoder_atts_list, dim=0) 
                targets = torch.cat(targets, dim=0)

                outputs = self.opt_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    return_dict=True,
                    labels=targets,
                )
                loss = outputs.loss 
            except:
                loss = 0
                for i in range(len(inputs_embeds_list)):
                    outputs = self.opt_model(
                        inputs_embeds=inputs_embeds_list[i],
                        attention_mask=encoder_atts_list[i],
                        return_dict=True,
                        labels=targets[i],
                    )
                    loss = loss + outputs.loss

            return {"loss": loss / len(inputs_embeds_list)}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["images"]
        prompts = samples["prompts"]
        
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()

        image_embeds = rearrange(
            image_embeds, "b v d -> 1 (b v) d"
        )
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            
        if self.instruct_qformer:
            text_Qformer = self.tokenizer(
                prompts,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output = query_output.last_hidden_state[:, :query_tokens.shape[1]]
            inputs_opt = self.opt_proj(query_output)
        else:

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_opt = self.opt_proj(query_output.last_hidden_state)
        

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
            image.device
        )

        opt_tokens = self.opt_tokenizer(
            prompts, padding="longest", return_tensors="pt"
        ).to(image.device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float):
            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)
            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        return output_text



    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        instruct_qformer = cfg.get("instruct_qformer", True)
        task_name = cfg.get("task_name", '')

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,

            instruct_qformer=instruct_qformer,
            task_name=task_name,
        )
        model.load_checkpoint_from_config(cfg)

        return model
