import click
import os


@click.command()
@click.option("--arch", "-a", default="cnn", type=str, help="cnn;transformer;mbc")
@click.option("--netid", "-n", default="hc856", type=str)
@click.option("--data_src", "-d", default="", type=str)
@click.option("--data_version", "-v", default=2, type=int)
@click.option("--cuda_id", "-c", default=0, type=int)
@click.option("--data_extra", "-ex", default="", type=str)
@click.option("--stitch", "-s", default=False, type=bool)
def main(arch, netid, data_src, data_version, cuda_id, data_extra, stitch):
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

    if arch == "cnn" or arch == "transformer":
        command = f"python train.py --config-dir=. --config-name=lowdim_pusht_diffusion_policy_{arch}.yaml"
        command += f" logging.name=train_diffusion_{arch}_{data_extra}_{stitch}"
        # if stitch:
        #     command += " task.dataset.enable_stitching=true"
        # else:
        #     command += " task.dataset.enable_stitching=false"

    # Common configurations
    command += f" dataloader.num_workers={num_workers}"
    command += f" dataloader.batch_size={batch_size}"
    command += f" val_dataloader.num_workers={num_workers}"
    command += f" val_dataloader.batch_size={batch_size}"
    command += " training.seed=42"
    command += f" training.device=cuda:{cuda_id}"
    command += f" hydra.run.dir='{data_src}/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}'"
    if data_extra:
        command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}_{data_extra}.zarr"
    else:
        # command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}.zarr"
        command += f" task.dataset.zarr_path={data_src}/pusht/pusht_cchi_v7_replay.zarr"
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()
