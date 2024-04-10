import click
import os


@click.command()
@click.option("--server_type", "-st", default="local", type=str, help="local or ilab")
@click.option("--netid", "-n", default="hc856", type=str)
@click.option("--data_src", "-d", default="", type=str)
@click.option("--control_type", "-ct", default="follow", type=str, help="repulse, region, follow")
@click.option("--integrate_type", "-it", default="concat", type=str, help="concat or controlnet")
@click.option("--cuda_id", "-c", default=0, type=int)
def main(server_type, netid, data_src, control_type, integrate_type, cuda_id):
    if server_type == "local":
        data_src = "./data"
    elif server_type == "ilab":
        data_src = f"/common/users/{netid}/Project/diffusion_policy/data"
    else:
        data_src = data_src

    command = f"python train.py --config-dir=. --config-name=image_pusht_control_diffusion_policy_cnn.yaml"
    command += f" policy.integrate_type='{integrate_type}'"
    command += " training.seed=42"
    command += f" training.device=cuda:{cuda_id}"
    command += f" hydra.run.dir='{data_src}/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}'"
    command += f" task.dataset.zarr_path={data_src}/kowndi_pusht_demo_v0_{control_type}.zarr"
    command += f" task.env_runner.control_type={control_type}"

    os.system(command)


if __name__ == "__main__":
    main()
