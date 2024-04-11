import click
import os


@click.command()
@click.option("--arch", "-a", default="transformer", type=str, help="cnn or transformer")
@click.option("--server_type", "-st", default="ilab", type=str, help="local or ilab")
@click.option("--netid", "-n", default="hc856", type=str)
@click.option("--data_src", "-d", default="", type=str)
@click.option("--control_type", "-ct", default="repulse", type=str, help="repulse, region, follow")
@click.option("--integrate_type", "-it", default="concat", type=str, help="concat or controlnet")
@click.option("--cuda_id", "-c", default=2, type=int)
def main(arch, server_type, netid, data_src, control_type, integrate_type, cuda_id):
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"
    else:
        data_src = data_src

    command = f"python train.py --config-dir=. --config-name=image_pusht_control_diffusion_policy_{arch}.yaml"
    command += f" policy.integrate_type='{integrate_type}'"
    command += " training.seed=42"
    command += f" training.device=cuda:{cuda_id}"
    command += f" hydra.run.dir='{data_src}/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}'"
    command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v0_{control_type}.zarr"
    command += f" task.env_runner.control_type={control_type}"
    command += f" logging.name=train_diffusion_{arch}_{control_type}_{integrate_type}"
    print(command)
    os.system(command)


if __name__ == "__main__":
    main()
