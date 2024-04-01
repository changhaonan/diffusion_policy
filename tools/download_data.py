import gdown
import os

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    download_key = "kowndi_pusht_demo_repulse_no_control"
    url_dict = {
        "kowndi_pusht_demo_repulse_no_control": "https://drive.usercontent.google.com/u/1/uc?id=1cVdygalbELc7DZYGJB1cu7bbYQ2tueRv&export=download",
    }
    url = url_dict[download_key]
    output = f"{data_dir}/{download_key}.zarr.tar.xz"

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        # Untar the file
        os.system(f"tar -xvf {output} -C {data_dir}")
