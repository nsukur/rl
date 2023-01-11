# Generate CSV adj and feature files from wsl files
# Needs podman container up with name "fermat"
# E.g.: podman run -v `pwd`:/rl/wd -it --name fermat fermat
# Finds all .wsl files in path and converts and saves them to the
# output dir

import subprocess
import os
from pathlib import Path
from os.path import join

dir_name = "hc-fit-mj-alpha-2020-06-final/hc-automator-18d-2020-06-final/"
adj_generator = "wsl-experiments/bin/adjacency-generator.wsl"
feature_generator = "wsl-experiments/bin/features-generator.wsl"
# base_command = ["docker", "exec", "fermat", "wsl"]
base_command = ["wsl"]


def convert_wsl_to_csv(path, graphs_dir="./graphs/"):
    full_out_path = join(graphs_dir, os.sep.join(Path(path).parts[:-1]))
    Path(full_out_path).mkdir(parents=True, exist_ok=True)

    p = Path(path)
    filename_adj = join(full_out_path, p.stem + "_adj.csv")
    filename_features = join(full_out_path, p.stem + "_features.csv")

    subprocess.call(
        base_command + [adj_generator, path, filename_adj],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.call(
        base_command + [feature_generator, path, filename_features],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return filename_adj


if __name__ == "__main__":
    counter = 0
    for path in Path(dir_name).rglob("*.wsl"):
        convert_wsl_to_csv(path)
        counter += 1

    print("Finished!", counter, "files processed")
