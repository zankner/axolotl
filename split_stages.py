import yaml
import argparse


if __name__ == "__main__":

    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-path", type=str, required=True)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.params_path, 'r') as file:
        data = yaml.safe_load(file)

    # Split the dictionary at the 'stage1' key
    stage1_data = data['stage_1']
    with open('stage_1.yml', 'w') as file:
        yaml.dump(stage1_data, file)

    # Split the dictionary at the 'stage2' key
    stage2_data = data['stage_2']
    with open('stage_2.yml', 'w') as file:
        yaml.dump(stage2_data, file)