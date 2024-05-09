from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", cache_dir="../pretrained_weights")
    snapshot_download("Rostlab/prot_bert", cache_dir="../pretrained_weights/protbert")
