from base_models.transformer import prot_model_id
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(prot_model_id, output_hidden_states=True)

prot = "MAFTFAAFCYMLTLVLCASLIFFVIWHIIAFDELRTDFKNPIDQGNPARARERLKNIERICCLLRKLVVPEYSIHGLFCLMFLCAAEWVTLGLNIPLLFYHLWRYFHRPADGSEVMYDAVSIMNADILNYCQKESWCKLAFYLLSFFYYLYSMVYTLVSF"

print(tokenizer(prot, return_tensors="pt", max_length=2048, padding=True, truncation=True))