import click
import os


@click.command()
@click.option("--arch", "-a", default="cnn", type=str, help="cnn or transformer")
@click.option("--netid", "-n", default="hc856", type=str)
@click.option("--data_src", "-d", default="", type=str)
@click.option("--data_version", "-v", default=1, type=int)
@click.option("--control_type", "-t", default="repulse", type=str, help="repulse, region, follow")
@click.option("--control_model", "-m", default="control_gate_unet", type=str, help="control_gate_unet, control_unet")
@click.option("--integrate_type", "-i", default="concat", type=str, help="concat or controlnet")
@click.option("--cfg_ratio", "-r", default=0.0, type=float)
@click.option("--cuda_id", "-c", default=0, type=int)
@click.option("--data_extra", "-ex", default="", type=str)
def main(arch, netid, data_src, data_version, control_type, control_model, integrate_type, cfg_ratio, cuda_id, data_extra):
    server_type = "local" if not os.path.exists("/common/users") else "ilab"
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"
    else:
        data_src = data_src

    command = f"python train.py --config-dir=. --config-name=image_pusht_control_diffusion_policy_{arch}.yaml"
    command += f" policy.control_model='{control_model}'"
    command += f" policy.integrate_type='{integrate_type}'"
    command += f" policy.cfg_ratio={cfg_ratio}"
    command += " training.seed=42"
    command += f" training.device=cuda:{cuda_id}"
    command += f" hydra.run.dir='{data_src}/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}'"
    if data_extra:
        command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}_{control_type}_{data_extra}.zarr"
    else:
        command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v{data_version}_{control_type}.zarr"
    command += f" task.env_runner.control_type={control_type}"
    command += f" logging.name=train_diffusion_{arch}_{control_type}_{integrate_type}_{data_extra}_{control_model}_{cfg_ratio}"
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()
