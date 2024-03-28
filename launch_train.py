import argparse
from mcli import RunConfig, create_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded", action="store_true")
    parser.add_argument("--size", type=str, required=True) 
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--priority", default="medium", choices=["high", "medium", "low", "lowest"])

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run = RunConfig.from_file("hydra_train.yaml")

    run.compute["cluster"] = args.cluster
    run.scheduling["priority"] = args.priority
    if args.preemptible:
        run.scheduling["resumable"] = True
    
    stage_1_yaml_name = f"{'hydra' if args.grounded else 'medusa'}_vicuna_{args.size}_qlora_stage1_sd_{args.seed}"
    stage_2_yaml_name = f"{'hydra' if args.grounded else 'medusa'}_vicuna_{args.size}_qlora_stage2_sd_{args.seed}"
    
    run.command = run.command.replace("{STAGE_1_NAME}", stage_1_yaml_name)
    run.command = run.command.replace("{STAGE_2_NAME}", stage_2_yaml_name)
    run.command = run.command.replace("{MODEL_SIZE}", args.size)

    if args.debug:
        with open("debug.yaml", "w") as f:
            f.write(str(run))
    else:
        launched_run = create_run(run)
        print(f"Launched: {launched_run.name}")