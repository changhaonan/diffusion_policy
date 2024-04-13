import gdown
import os


def download_dataset(data_src, download_key: str):
    print(f"Start download {download_key}...")
    url_dict = {
        f"kowndi_pusht_demo_v0_repulse": "https://drive.google.com/file/d/1XUG6f9qKxwG8lDJ3r64Y36X4L8Lpz-HJ/view?usp=sharing",
        f"kowndi_pusht_demo_v0_follow": "https://drive.google.com/file/d/1AohGsWMvZuaYsFM4slEEL5MO7LC9zQO0/view?usp=sharing",
        f"kowndi_pusht_demo_v0_region": "https://drive.google.com/file/d/1DkKaj-t5GF2inKlpJQ32pT7_UpxtKBSr/view?usp=sharing",
        f"kowndi_pusht_demo_v1_repulse": "https://drive.google.com/file/d/1LOywDZMYe6N_rmeBVSE0Ywm2TjdQ9rQU/view?usp=sharing",
    }
    url = url_dict[download_key]
    output = f"{data_src}/{download_key}.zarr.tar.xz"

    os.makedirs(data_src, exist_ok=True)
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True)
        if output.endswith(".tar.xz"):
            # Untar the file
            os.system(f"tar -xvf {output} -C {data_src}")
            # Remove the tar file
            os.remove(output)


if __name__ == "__main__":
    import os

    data_version = 1
    server_type = "local" if not os.path.exists("/common/users") else "ilab"
    netid = "hc856"
    control_type = "repulse"
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"

    ## Download the dataset
    download_key = f"kowndi_pusht_demo_v{data_version}_{control_type}"
    download_dataset(data_src, download_key)
