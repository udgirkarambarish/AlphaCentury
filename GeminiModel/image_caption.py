import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import json
cache_dir = r"E:\sih"  # The path you provided
os.makedirs(cache_dir, exist_ok=True)
# caption_cache_file = "image_captions_cache.json"
device=torch.device('cpu')
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224",cache_dir=cache_dir).to(device)
torch.set_num_threads(12)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224",cache_dir=cache_dir)
# if os.path.exists(caption_cache_file):
# #     with open(caption_cache_file, "r") as f:
# #         captions_cache = json.load(f)
# else:
captions_cache = {}
def generate_image_caption(image_path, prompt="<grounding>Describe the condition and objects in this scene in detail."):
    #model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    #processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    image_name = os.path.basename(image_path)
    if image_name in captions_cache:
        return captions_cache[image_name]


    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate Caption
    generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
            temperature=0.1 
    )
   


    # Decode Output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    captions_cache[image_name] = generated_text.strip()
    # with open(caption_cache_file, "w") as f:
    #     json.dump(captions_cache, f, indent=4)
    
    return generated_text.strip()

def generate_ene_caption(image_path, prompt="<grounding>Describe the overall environment, including objects, conditions, and any notable features."):
    #model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    #processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    image_name = os.path.basename(image_path)
    if image_name in captions_cache:
        return captions_cache[image_name]

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Generate Caption
    generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
            temperature=0.1
    )

    # Decode Output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    captions_cache[image_name] = generated_text.strip()
    # with open(caption_cache_file, "w") as f:
    #     json.dump(captions_cache, f, indent=4)
    return generated_text.strip()

# def generate_ene2_caption(image_path, prompt="<grounding>Describe the overall environment, including objects, conditions, and any notable features. after changing the enviroment state"):
#     #model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
#     #processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
#     image_name = os.path.basename(image_path)
#     if image_name in captions_cache:
#         return captions_cache[image_name]

#     image = Image.open(image_path).convert("RGB").resize((224, 224))
#     inputs = processor(text=prompt, images=image, return_tensors="pt")

#     # Generate Caption
#     generated_ids = model.generate(
#             pixel_values=inputs["pixel_values"],
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             image_embeds=None,
#             image_embeds_position_mask=inputs["image_embeds_position_mask"],
#             use_cache=True,
#             max_new_tokens=128,
#             temperature=0.1 
#     )

#     # Decode Output
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     captions_cache[image_name] = generated_text.strip()
#     with open(caption_cache_file, "w") as f:
#         json.dump(captions_cache, f, indent=4)
#     return generated_text.strip()