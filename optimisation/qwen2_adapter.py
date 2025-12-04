from .base_adapter import BaseModelAdapter
from PIL import Image
from .utils import apply_random_patch, semantic_similarity_loss
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import math

import torch
import torch.nn.functional as F
import io
import base64

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
class Qwen2Adapter(BaseModelAdapter):
    def load(self, model_id):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto"
            )
            print("initialising qwen2.5 model!!!  ")
        
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto"
            )
        return processor, model
    
    # just do it one at a time first
    def process_target(self, image_path, prompt_text, target_text, keywords = []):
        """
        Given an image path, prompt, and target string, generate input_ids, labels, and image tensor.
        """
        
        # Load and preprocess the image
        if image_path:
            with Image.open(image_path) as img:
                # resize to 336 for now
                if self.patch_only:
                    resized_img = img.resize(self.patch_size)
                else:
                    resized_img = img.resize(self.image_size)

                buffer = io.BytesIO()
                resized_img.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Process image and full prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        processed = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        input_ids = processed["input_ids"]
    
        attention_mask = processed.get("attention_mask", None)
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed['image_grid_thw']

        # input only
        if not target_text:
            return input_ids, None, pixel_values, image_grid_thw, attention_mask

        target_lines = target_text.split('\n')
        #print(target_lines)
        target_lines_ids = []
        target_lines_labels = [] 

        # note: this is only effective for targets in the form of incomplete lists
        for line in target_lines:
            processed_line = self.processor(text=line + "\n\n", return_tensors="pt")["input_ids"].to('cuda')
            
            target_lines_ids.append(processed_line.clone())
            # Set the last token to -100 to ignore it in loss computation
            processed_line[0, -1] = -100
            target_lines_labels.append(processed_line)

        
        # Construct labels and full input
        target_labels = torch.cat(target_lines_labels, dim=1)
        labels = torch.full_like(input_ids, -100)
        labels = torch.cat([labels, target_labels], dim=1)
            
        target_ids = torch.cat(target_lines_ids, dim=1)
        input_ids = torch.cat([input_ids, target_ids], dim=1)

        # Extend attention mask for target tokens
        target_attention = torch.ones_like(target_ids, device='cuda')  # All 1s for target tokens
        attention_mask = torch.cat([attention_mask, target_attention], dim=1)


        # Optionally compute manual weights for tokens: 0s for prompt, decreasing for target
        # note this is not used in the final method
        prompt_len = input_ids.size(1) - target_ids.size(1)
        weights = torch.cat([
            torch.zeros(prompt_len, device='cuda'),
            torch.linspace(1, 0, steps=target_ids.size(1), device='cuda')
        ], dim=0)

        for i, tid in enumerate(target_ids[0]):
            token_str = self.processor.tokenizer.decode([tid.item()]).lower()
            if "".join(char for char in token_str if char.isalpha()) in keywords:
                weights[prompt_len + i] += 1.0

        return input_ids, labels, pixel_values, image_grid_thw, attention_mask, weights

    def preprocess_patched(self, patched_tensor, image_grid_thw, temporal_patch_size=2, patch_size=14, merge_size=2):
        (grid_t, grid_h, grid_w) = image_grid_thw

        # pytorch version 
        patches = patched_tensor.unsqueeze(0)  # Add dimension

        channel = patches.shape[1]
        
        T = patches.size(0)
        remainder = T % temporal_patch_size
        if remainder != 0:
            num_to_pad = temporal_patch_size - remainder
            last_patch = patches[-1:]  # shape (1, ...)
            pad_tensor = last_patch.expand(num_to_pad, *last_patch.shape[1:])  # expand keeps it differentiable
            patches = torch.cat([patches, pad_tensor], dim=0)

        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )

        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # Match np.transpose

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches

    def apply_patch(self, image_tensor_batch, image_grid_thw, patch):
        B, N, D = image_tensor_batch.shape

        # Remove extra batch dimension from patch if necessary
        while patch.dim() > 3:
            patch = patch.squeeze(0)

        if self.patch_only:
            empty_img = torch.full((3, self.patch_size[0], self.patch_size[1]), -1.0, device=patch.device, dtype=patch.dtype, requires_grad=True)
        else:
            empty_img = torch.full((3, self.image_size[0], self.image_size[1]), -1.0, device=patch.device, dtype=patch.dtype, requires_grad=True)

        #patched_img = apply_patch_to_image( empty_img, patch, top, left)

        if self.patch_only:
            #patched_img = apply_random_patch( empty_img, patch, scale_range=(1.0, 1.0), rotation_range=(0, 0))
            transformed_patch = self.preprocess_patched(patch, image_grid_thw)
            return transformed_patch.unsqueeze(0).expand(B, -1, -1) 
     
        patched_img = apply_random_patch( empty_img, patch, scale_range=(0.8, 1.2), rotation_range=(-15, 15))       
        transformed_patch = self.preprocess_patched(patched_img, image_grid_thw)

        # Create a mask for regions not equal to -1
        mask = (transformed_patch != -1.0).float()  # shape (N, D)

        # Expand to batch
        transformed_patch_batch = transformed_patch.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        mask_batch = mask.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Blend into the original batch
        patched_batch = image_tensor_batch * (1 - mask_batch) + transformed_patch_batch * mask_batch

        return patched_batch
    

    # cross entropy loss
    def loss_function(self, logits, labels, weights=None):
        # !! shifts
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
        
        mask = (labels != -100).float()
        if weights is not None:
            weights = weights[:, 1:].reshape(-1)
            mask = mask * weights
    
        loss = (loss * mask).sum() / mask.sum()
        return loss

        # optionally weigh tokens in CE loss computation
        # this is not used in final method

        # if weights is not None:
        #     #weights = weights[:, 1:].contiguous().view(-1)  # same shift as labels
        #     weights = weights[:, 1:]
        #     weights = weights.contiguous().view(-1)
        #     # print(loss.shape)
        #     # print(weights.shape)
        #     loss = loss * weights
        # return loss.mean()

    def compute_loss(self, target, patch, suffix='', custom_loss=True, print_probs=False):

        input_ids, labels, pixel_values, image_grid_thw, attention_mask, weights = self.process_target(
            target['image'],
            target['text'] + suffix,
            target['target'],
            target['keywords'].split(' ')
        )
        embedding_layer = self.model.get_input_embeddings() 
        embedding_matrix = embedding_layer.weight
       
        # normalise patch  (for patch only mode?)
        if self.patch_only:
            mean_tensor = torch.tensor(OPENAI_CLIP_MEAN).view(-1, 1, 1).to(self.device)
            std_tensor = torch.tensor(OPENAI_CLIP_STD).view(-1, 1, 1).to(self.device)
            normalised_patch = (patch - mean_tensor) / std_tensor 
            patched_imgs = self.apply_patch(pixel_values.unsqueeze(0), image_grid_thw[0], normalised_patch)
        else:
            patched_imgs = self.apply_patch(pixel_values.unsqueeze(0), image_grid_thw[0], patch)
             
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=patched_imgs,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        if print_probs:
            print(target['target'])

            if self.patch_only:
                print(patched_imgs)

            self.log_topk(outputs, labels)

        if custom_loss:
            logits = outputs.logits 
            loss = semantic_similarity_loss(logits, labels, embedding_matrix, weights = weights.unsqueeze(0), mode="attention", verbose=print_probs)

        else:
            logits = outputs.logits 
            loss = self.loss_function(logits, labels, weights=None)
        return loss

    

    def generate(self, prompt, image_path):
        if image_path:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")


        # Generate with scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Decode final text
        decoded_output = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
      
        return decoded_output

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
        


