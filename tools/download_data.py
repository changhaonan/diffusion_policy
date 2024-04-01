import gdown
import os

if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    download_key = "kowndi_pusht_demo_repulse_no_control"
    url_dict = {
        "kowndi_pusht_demo_repulse_no_control": "https://drive.google.com/file/d/1ne_dEVh599e7hgHZ6sPcggR7omgBjTEK/view?usp=drive_link",
    }
    url = url_dict[download_key]
    output = f"{root_dir}/{download_key}.zarr"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
