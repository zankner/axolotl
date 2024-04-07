import argparse
from mcli import RunConfig, create_run

GLOBAL_BATCH_SIZE = 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded", action="store_true")
    parser.add_argument("--size", type=str, required=True) 
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--head-arch", type=str, required=True)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--device-batch-size", type=int, required=True)
    parser.add_argument("--priority", default="medium", choices=["high", "medium", "low", "lowest"])

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    run = RunConfig.from_file("hydra_train.yaml")

    run.compute["cluster"] = args.cluster
    run.scheduling["priority"] = args.priority
    if args.preemptible:
        run.scheduling["resumable"] = True
    
    # Build the run name
    run_name_base = f"{'hydra' if args.grounded else 'medusa'}_vicuna_{args.size}_{args.head_arch}_nh_{args.num_heads}_lmw_{args.lm_weight}_kdw_{args.kd_weight}"
    stage_1_name = f"{run_name_base}_stage1_sd_{args.seed}"
    stage_2_name = f"{run_name_base}_stage2_sd_{args.seed}"
    run.name = run_name_base

    # Set the train command    
    run.command = run.command.replace("{STAGE_1_NAME}", stage_1_name)
    run.command = run.command.replace("{STAGE_2_NAME}", stage_2_name)

    # Set parameters
    run.parameters["stage_1"]["seed"] = args.seed
    run.parameters["stage_2"]["seed"] = args.seed

    run.parameters["stage_1"]["wandb_name"] = stage_1_name
    run.parameters["stage_2"]["wandb_name"] = stage_2_name

    assert GLOBAL_BATCH_SIZE % (args.device_batch_size * 8) == 0, "Global batch size must be divisible by device batch size * 8"
    run.parameters["stage_1"]["gradient_accumulation_steps"] = GLOBAL_BATCH_SIZE // (args.device_batch_size * 8)
    run.parameters["stage_2"]["gradient_accumulation_steps"] = GLOBAL_BATCH_SIZE // (args.device_batch_size * 8)
    run.parameters["stage_1"]["micro_batch_size"] = args.device_batch_size
    run.parameters["stage_2"]["micro_batch_size"] = args.device_batch_size

    run.parameters["stage_1"]["grounded_heads"] = args.grounded
    run.parameters["stage_2"]["grounded_heads"] = args.grounded

    run.parameters["stage_1"]["hydra_head_arch"] = args.head_arch
    run.parameters["stage_2"]["hydra_head_arch"] = args.head_arch

    run.parameters["stage_1"]["hydra_num_heads"] = args.num_heads
    run.parameters["stage_2"]["hydra_num_heads"] = args.num_heads

    if args.debug:
        with open("debug.yaml", "w") as f:
            f.write(str(run))
    else:
        launched_run = create_run(run)
        print(f"Launched: {launched_run.name}")