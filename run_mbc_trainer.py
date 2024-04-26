import click
import os


@click.command()
@click.option("--arch", "-a", default="mbc", type=str, help="mbc")
@click.option("--netid", "-n", default="hc856", type=str)
@click.option("--data_src", "-d", default="", type=str)
@click.option("--data_version", "-v", default=2, type=int)
@click.option("--control_type", "-t", default="repulse", type=str, help="repulse, region, follow")
@click.option("--cuda_id", "-c", default=0, type=int)
@click.option("--data_extra", "-ex", default="", type=str)
@click.option("--stitch", "-s", default=False, type=bool)
@click.option("--max_modality", "-k", default=4, type=int)
def main(arch, netid, data_src, data_version, control_type, cuda_id, data_extra, stitch, max_modality):
    server_type = "local" if not os.path.exists("/common/users") else "ilab"
    if server_type == "local":
        data_src = "./data"
        num_workers = 8
        batch_size = 64
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"
        num_workers = 12
        batch_size = 64
    else:
        data_src = data_src
        num_workers = 8
        batch_size = 64

    if arch == "mbc":
        command = f"python train.py --config-dir=. --config-name=lowdim_pusht_mbc_policy.yaml"
        command += f" logging.name=train_mbc_{control_type}_{data_extra}_{max_modality}"

    # Common configurations
    command += f" policy.n_max_modality={max_modality}"
    command += f" dataloader.num_workers={num_workers}"
    command += f" dataloader.batch_size={batch_size}"
    command += f" val_dataloader.num_workers={num_workers}"
    command += f" val_dataloader.batch_size={batch_size}"
    command += " training.seed=42"
    command += f" training.device=cuda:{cuda_id}"
    command += f" hydra.run.dir='{data_src}/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}'"
    if data_extra:
        command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}_{control_type}_{data_extra}.zarr"
    else:
        command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}_{control_type}.zarr"
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()
