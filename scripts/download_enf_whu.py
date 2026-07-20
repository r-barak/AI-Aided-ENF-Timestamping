import argparse
import requests
from pathlib import Path
from tqdm import tqdm


def fetch_file_list(api_url):
    """
    Recursively crawls the GitHub API to build a flat list of all files and their relative paths.
    """
    file_list = []
    response = requests.get(api_url)
    
    if response.status_code == 403:
        print("\nError: GitHub API rate limit exceeded. Try again later.")
        return []
    response.raise_for_status()
    
    items = response.json()
    
    for item in items:
        if item['type'] == 'file':
            file_list.append({
                'download_url': item['download_url'],
                'file_name': item['name']
            })
            
    return file_list


def download_files(files, output_dir):
    """
    Downloads a list of files to the target directory with a single progress bar.
    """
    if not files:
        print("No files found to download.")
        return

    for file_info in tqdm(files, desc="Downloading Dataset", unit="file"):
        file_url = file_info['download_url']
        final_file_path = output_dir / file_info['file_name']
        final_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_response = requests.get(file_url)
        with open(final_file_path, 'wb') as f:
            f.write(file_response.content)


def main():
    parser = argparse.ArgumentParser(
        description="Download the ENF_WHU Dataset (H1 subset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--project_dir', 
        type=str, 
        default='.', 
        metavar='<path>',
        help='The root directory of the project'
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='ENF_WHU', 
        metavar='<name>',
        help='Name of the dataset folder to create'
    )

    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    data_dir = project_dir / 'data' / 'datasets' / args.dataset_name / '01_raw'
    api_urls = {
        'query': 'https://api.github.com/repos/ghua-ac/ENF-WHU-Dataset/contents/ENF-WHU-Dataset/H1',
        'reference': 'https://api.github.com/repos/ghua-ac/ENF-WHU-Dataset/contents/ENF-WHU-Dataset/H1_ref'
    }

    for data_type, api_url in api_urls.items():
        output_dir = data_dir / f'{data_type}_recordings'
        print(f"Fetching {data_type} files list...")
        files = fetch_file_list(api_url)
        print(f"Found {len(files)} {data_type} files.")
        print(f"Downloading {data_type} files to: {output_dir}")
        download_files(files, output_dir)
        print("\nDownload complete!")

if __name__ == "__main__":
    main()