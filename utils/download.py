from huggingface_hub import snapshot_download
def download_model(repo_id, local_dir):
    snapshot_download(repo_id, local_dir=local_dir, resume_download=True)
