import os
import json
from huggingface_hub import hf_hub_download, HfApi, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError

def load_hub_data(repo_id, filename, extension="json"):
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    if extension.lower() == "json":
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    elif extension.lower() == "txt":
        with open(json_path, 'r') as f:
            data = [line.strip() for line in f]
        return data
    else:
        print(f"Unsupported extension: {extension}.")
        return None

def upload_file(file_path, username, repo_id, repo_type="dataset", commit_message="Upload file"):
    """
    Uploads a single file to a Hugging Face Hub repository. Creates the repository
    if it doesn't exist using the API.

    Args:
        file_path (str): The path to the local file to upload.
        repo_id (str): Your Hugging Face username and the desired repository name
                         (e.g., "your_username/my_dataset" or "your_username/my_model").
        repo_type (str, optional): The type of repository ("dataset" or "model").
                                   Defaults to "dataset".
        commit_message (str, optional): The commit message for the upload.
                                       Defaults to "Upload file".
    """
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repository '{repo_id}' of type '{repo_type}' created or already exists.")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=f"{username}/{repo_id}",
            repo_type=repo_type,
            commit_message=commit_message
        )
        print(f"Successfully uploaded '{os.path.basename(file_path)}' to Hugging Face Hub {repo_type} repository: {repo_id}")
    except RepositoryNotFoundError:
        print(f"Error: Repository '{repo_id}' not found. Please ensure the repository ID is correct.")
    except Exception as e:
        print(f"An error occurred during file upload: {e}")
        print("Please ensure you are logged in using 'huggingface-cli login'.")

def upload_folder(folder_path, username, repo_id, repo_type="dataset", commit_message="Upload folder"):
    """
    Uploads an entire folder to a Hugging Face Hub repository. Creates the repository
    if it doesn't exist using the API.

    Args:
        folder_path (str): The path to the local folder to upload.
        repo_id (str): Your Hugging Face username and the desired repository name
                         (e.g., "your_username/my_dataset" or "your_username/my_model").
        repo_type (str, optional): The type of repository ("dataset" or "model").
                                   Defaults to "dataset".
        commit_message (str, optional): The commit message for the upload.
                                       Defaults to "Upload folder".
    """
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repository '{repo_id}' of type '{repo_type}' created or already exists.")
        upload_folder(
            folder_path=folder_path,
            repo_id=f"{username}/{repo_id}",
            repo_type="dataset",
            commit_message="Upload navigable data for R2R dataset"
        )
        print(f"Successfully uploaded folder '{folder_path}' to Hugging Face Hub {repo_type} repository: {repo_id}")
    except RepositoryNotFoundError:
        print(f"Error: Repository '{repo_id}' not found. Please ensure the repository ID is correct.")
    except Exception as e:
        print(f"An error occurred during folder upload: {e}")
        print("Please ensure you are logged in using 'huggingface-cli login'.")