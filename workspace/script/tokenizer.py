from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("../tokenizer.json")
tokenizer.enable_padding(length=32, pad_id=49407)
tokenizer.enable_truncation(max_length=32)


prompt_text = "hand"
encoded = tokenizer.encode(prompt_text)
print("ids : ", encoded.ids)
print("attention_mask : ", encoded.attention_mask)