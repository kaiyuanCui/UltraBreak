# adapters/llava_adapter.py
from .base_adapter import BaseModelAdapter
from PIL import Image
import torch
from .utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, apply_random_patch, semantic_similarity_loss
import torch.nn.functional as F
import io, base64
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

import math

class Llava16Adapter(BaseModelAdapter):
    def load(self, model_id):
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        return processor, model

    def process_target(self, image_path, prompt_text, target_text, keywords=[]):
        """
        Prepares inputs for LLaVA 1.6 model.
        """
        full_prompt = f"USER: <image>\n{prompt_text}\n\nASSISTANT:"

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        if self.patch_only:
            resized_img = image.resize(self.patch_size)
        else:
            resized_img = image.resize(self.image_size)
        processed = self.processor(images=resized_img, text=full_prompt, return_tensors="pt").to("cuda")
        image_sizes = processed['image_sizes']
        input_ids = processed["input_ids"]
        attention_mask = processed.get("attention_mask", None)
        pixel_values = processed["pixel_values"]
        # Process target text line by line
        target_lines = target_text.split("\n")
        target_ids, target_labels = [], []

        # note: this is only effective for targets in the form of incomplete lists
        for line in target_lines:
            proc_line = self.processor(text=line + "\n", return_tensors="pt")["input_ids"].to("cuda")
            target_ids.append(proc_line.clone())

            proc_line[0, -1] = -100  # mask last token
            target_labels.append(proc_line)

        target_ids = torch.cat(target_ids, dim=1)
        target_labels = torch.cat(target_labels, dim=1)

        labels = torch.full_like(input_ids, -100)
        labels = torch.cat([labels, target_labels], dim=1)
        input_ids = torch.cat([input_ids, target_ids], dim=1)

        # Extend attention mask
        target_attention = torch.ones_like(target_ids, device="cuda")
        attention_mask = torch.cat([attention_mask, target_attention], dim=1)

        # Optionally compute manual weights for tokens: 0s for prompt, decreasing for target
        # note this is not used in the final method
        prompt_len = input_ids.size(1) - target_ids.size(1)
        weights = torch.cat([
            torch.zeros(prompt_len, device="cuda"),
            torch.ones(target_ids.size(1), device="cuda")
        ], dim=0)

        for i, tid in enumerate(target_ids[0]):
            token_str = self.processor.tokenizer.decode([tid.item()]).lower()
            if any(word in token_str for word in keywords):
                weights[prompt_len + i] = 10.0


    def apply_patch(self, image_tensor_batch, patch):
        """
        Apply adversarial patch directly on pixel space.
        """
        B, P, C, H, W = image_tensor_batch.shape
        patched_imgs = []

        # optionally normalise patch first
        # mean_tensor = torch.tensor(OPENAI_CLIP_MEAN).view(-1, 1, 1).to('cuda')
        # std_tensor = torch.tensor(OPENAI_CLIP_STD).view(-1, 1, 1).to('cuda')
        # normalised_patch = (patch - mean_tensor) / std_tensor 
        normalised_patch = patch
        added = 0
        
        for img in image_tensor_batch[0]:
            # apply the adversarial patch to only the first preprocessed image as an approximation
            # better results may be obtained by fully mimicking the image preprocessing of llava
            if added < 1:
                patched = apply_random_patch(img, normalised_patch)
                patched_imgs.append(patched)
                added += 1
            else:
                patched_imgs.append(img)

        return torch.stack(patched_imgs, dim=0).unsqueeze(0)
        
    
    def loss_function(self, logits, labels, weights=None):
        logits = logits[:, :-1, :]  # [B, T-1, V]
        labels = labels[:, 1:]  # [B, T-1]
    
        logits = logits.contiguous().view(-1, logits.size(-1))  # [B*T, V]
        labels = labels.contiguous().view(-1)                   # [B*T]

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index= -100,
            reduction='none'
        )
        if weights is not None:
            weights = weights[:, 1:]
            weights = weights.contiguous().view(-1)
            loss = loss * weights
        return loss.mean()

    

    def compute_loss(self, target, patch, suffix="", custom_loss=True, print_probs=False):
        input_ids, labels, pixel_values, _, attention_mask, weights, image_sizes = self.process_target(
            target['image'],
            target['text'],
            target['target'],
            target['keywords'].split(' ')
        )
        multiplier = 1.0 if target['target_type'] == 'positive' else -1.0
        patched_imgs = self.apply_patch(pixel_values, patch)
        #print('patched shape:')
        #print(patched_imgs.shape)

        
        if not custom_loss:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=patched_imgs,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                labels=labels
            )
            logits = outputs.logits
            loss = self.loss_function(logits, labels, weights.unsqueeze(0))
        else:
            embedding_layer = self.model.get_input_embeddings() 
            embedding_matrix = embedding_layer.weight
            outputs = self.model(
                    #inputs_embeds=inputs_embed,
                    input_ids=input_ids,
                    pixel_values=patched_imgs,
                    attention_mask=attention_mask,
                    image_sizes=image_sizes,
                    #labels=labels,
                )
            
            if print_probs:
                print(target['target'])
                if self.patch_only:
                    print(patched_imgs)
                self.log_topk(outputs, labels)
    
           
            logits = outputs.logits 
            loss = semantic_similarity_loss(logits, labels, embedding_matrix, weights = weights.unsqueeze(0), mode="attention", verbose=print_probs)
            #loss = self.loss_function(logits, labels, weights.unsqueeze(0)) * multiplier
            return loss
        return loss

    def generate(self, prompt, image_path):
        prompt = f"[INST] <image>\n{prompt}[/INST]"
        if image_path:
            print(image_path)
            image = Image.open(image_path)
        else:
            image = None
        inputs = self.processor(image, prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(output[0], skip_special_tokens=True)
        print(response)

    def log_topk(self, outputs, labels=None, top_k=5):
        """
        Logs top-k tokens with probabilities for each position in the forward pass.
        
        Args:
            outputs: Model output (must contain logits)
            tokenizer: Tokenizer to decode token IDs
            top_k: Number of top tokens to show per position
        """
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
        labels = labels[:, 1:]  # [B, T-1]
    
        batch_size, seq_len, vocab_size = logits.shape
    
        # Apply softmax across vocab dimension
        probs = F.softmax(logits, dim=-1)
    
        for pos in range(seq_len):
            
            if labels is not None:
                # Skip positions where label == -100
                if pos<seq_len-1 and labels[0, pos].item() == -100:
                    continue
                
            pos_probs = probs[0, pos]  # first sample, position `pos`
            top_probs, top_indices = pos_probs.topk(top_k)
            top_tokens = [self.processor.tokenizer.decode([idx]) for idx in top_indices.tolist()]
    
            print(f"Position {pos}:")
            for token, prob in zip(top_tokens, top_probs.tolist()):
                print(f"  {token}: {prob:.4f}")
            print("-" * 40)

