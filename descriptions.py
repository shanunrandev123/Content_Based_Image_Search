import torch
from datasets import load_dataset

from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_4bit=True, torch_dtype=torch.bfloat16)

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "train"),
    ("Mattijs/snacks", None, "validation"),
    ("rokmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train")

]

prompt1 = "describe this image in full detail. Describe each and every aspect of the image"

prompt2 = "create an extensive description of this image"


counter = 0
for name, config, split in datasets:
    dataset = load_dataset(name, config, split=split)
    for idx in range(len(dataset)):
        image = dataset[idx]["image"]
        desc = ""

        for _prompt in [prompt1, prompt2]:
            inputs = processor(image, text=_prompt, return_tensors="pt").to(model.device, torch.bfloat16)
            outputs = model.generate(**inputs, do_sample=False, temperature=1, max_length=512, min_length=16, top_p=0.9)
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            desc += generated_text + " "
        desc = desc.strip()
        image.save(f"images/{counter}.jpg")

        print(counter, desc)

        with open("description.csv", "a") as f:
            f.write(f"{counter},{desc}\n")

        counter += 1
        torch.cuda.empty_cache()




