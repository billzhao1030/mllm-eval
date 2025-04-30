from huggingface_util import upload_file

if __name__ == "__main__":
    username = "billzhao1030"
    scan_file = '../datasets/R2R/connectivity/scans.txt'
    repo_id = f"MP3D"

    # Upload the entire navigable file to Hugging Face
    upload_file(scan_file, username, repo_id)

    print("Script finished: Scan data uploaded to Hugging Face Hub.")