# import argparse
# import requests
# from pathlib import Path
# from tqdm import tqdm


# def fetch_file_list(api_url, current_subpath=""):
#     """
#     Recursively crawls the GitHub API to build a list of all files,
#     preserving their relative folder structure.
#     """
#     file_list = []
#     response = requests.get(api_url)
    
#     if response.status_code == 403:
#         print("\nError: GitHub API rate limit exceeded. Try again later.")
#         return []
#     response.raise_for_status()
    
#     items = response.json()
    
#     for item in items:
#         if item['type'] == 'file' and item['name'].endswith('.mat'):
#             # Append the filename to whatever subfolder we are currently in
#             rel_path = Path(current_subpath) / item['name']
#             file_list.append({
#                 'download_url': item['download_url'],
#                 'relative_path': str(rel_path)
#             })
#         elif item['type'] == 'dir':
#             # If it's a directory, trigger the recursion using the directory's API URL
#             sub_url = item['url']
#             sub_path = Path(current_subpath) / item['name']
#             file_list.extend(fetch_file_list(sub_url, str(sub_path)))
            
#     return file_list


# def download_files(files, output_dir):
#     """
#     Downloads files to the target directory, recreating nested folders as needed.
#     Skips files that have already been downloaded.
#     """
#     if not files:
#         print("No files found to download (List is empty).")
#         return

#     for file_info in tqdm(files, desc="Downloading Dataset", unit="file"):
#         file_url = file_info['download_url']
#         final_file_path = output_dir / file_info['relative_path']
#         final_file_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # --- NEW CODE: SKIP IF FILE ALREADY EXISTS ---
#         if final_file_path.exists():
#             continue 
#         # ---------------------------------------------

#         file_response = requests.get(file_url)
#         with open(final_file_path, 'wb') as f:
#             f.write(file_response.content)

# def main():
#     parser = argparse.ArgumentParser(
#         description="Download the ENF_WHU Dataset (H1 subset)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         '--project_dir', 
#         type=str, 
#         default='.', 
#         metavar='<path>',
#         help='The root directory of the project'
#     )
#     parser.add_argument(
#         '--dataset_name', 
#         type=str, 
#         default='ENF_BGU', 
#         metavar='<name>',
#         help='Name of the dataset folder to create'
#     )

#     args = parser.parse_args()

#     project_dir = Path(args.project_dir).resolve()
#     data_dir = project_dir / 'data' / 'datasets' / args.dataset_name / '01_raw'
#     api_url = "https://api.github.com/repos/r-barak/ENF-BGU/contents"

#     print(f"Fetching files list...")
#     files = fetch_file_list(api_url)
#     print(f"Found {len(files)} files.")
#     print(f"Downloading files to: {data_dir}")
#     download_files(files, data_dir)
#     print("\nDownload complete!")

# if __name__ == "__main__":
#     main()

import argparse
import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def get_latest_release_url(repo_owner, repo_name):
    """
    Asks the GitHub API for the latest release of a repository and 
    extracts the direct download link for the attached zip file.
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
    
    print(f"Checking for the latest dataset release...")
    response = requests.get(api_url)
    
    if response.status_code == 404:
        print("Error: No releases found for this repository.")
        return None
        
    response.raise_for_status()
    release_data = response.json()
    
    # Releases can have multiple files attached. We assume you only uploaded one zip.
    assets = release_data.get('assets', [])
    if not assets:
        print("Error: The latest release has no files attached to it.")
        return None
        
    # Grab the download URL of the first attached file
    download_url = assets[0]['browser_download_url']
    file_name = assets[0]['name']
    
    print(f"Found release: {release_data['tag_name']} ({file_name})")
    return download_url


def download_and_extract(zip_url, output_dir):
    """
    Downloads a large zip file with a progress bar and extracts it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_zip_path = output_dir / "temp_dataset.zip"

    print(f"Connecting to download server...")
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 

    print(f"Downloading dataset...")
    with open(temp_zip_path, 'wb') as file, tqdm(
        desc="Progress",
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    print("\nExtracting files...")
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print("Cleaning up temporary files...")
    os.remove(temp_zip_path)
    print(f"Success! Dataset extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the latest ENF_BGU dataset release",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--project_dir', type=str, default='.', help='Root directory')
    parser.add_argument('--dataset_name', type=str, default='ENF_BGU', help='Dataset folder name')

    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    data_dir = project_dir / 'data' / 'datasets' / args.dataset_name / '01_raw'

    # 1. Dynamically get the link to the newest zip file
    repo_owner = "r-barak"
    repo_name = "ENF-BGU"
    zip_url = get_latest_release_url(repo_owner, repo_name)
    
    # 2. If a link was successfully found, download and extract it
    if zip_url:
        download_and_extract(zip_url, data_dir)

if __name__ == "__main__":
    main()