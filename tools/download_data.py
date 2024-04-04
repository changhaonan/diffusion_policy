import gdown
import os

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    download_key = "kowndi_pusht_demo_v0_follow"
    url_dict = {
        "kowndi_pusht_demo_repulse_no_control": "https://drive.google.com/file/d/1cVdygalbELc7DZYGJB1cu7bbYQ2tueRv/view?usp=sharing",
        "kowndi_pusht_demo_v0_repulse": "https://drive.google.com/file/d/1XUG6f9qKxwG8lDJ3r64Y36X4L8Lpz-HJ/view?usp=sharing",
        "kowndi_pusht_demo_v0_follow": "https://drive.google.com/file/d/1AohGsWMvZuaYsFM4slEEL5MO7LC9zQO0/view?usp=sharing"
    }
    url = url_dict[download_key]
    output = f"{data_dir}/{download_key}.zarr.tar.xz"

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True)
        if output.endswith(".tar.xz"):
            # Untar the file
            os.system(f"tar -xvf {output} -C {data_dir}")
            # Remove the tar file
            os.remove(output)
