from collections import OrderedDict
import json

def load_config(file_path):
    json_str = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str)
    return config

def parse(args):
    config = load_config(args.config)
    config["image_folder_path"] = config["pipeline"]["image_folder_path"]
    config["result_folder_path"] = config["pipeline"]["result_folder_path"]
    config["batch_size"] = config["pipeline"]["batch_size"]
    config["image_size"] = config["pipeline"]["image_size"]
    config["anomalib_config_path"] = config["pipeline"]["anomalib"]["config_path"]
    config["anomalib_checkpoint_path"] = config["pipeline"]["anomalib"]["checkpoint_path"]
    config["anomalib_prediction_threshold"] = config["pipeline"]["anomalib"]["prediction_threshold"]
    config["n_noising_step"] = config["pipeline"]["heatmap"]["n_noising_step"]
    config["n_iter"] = config["pipeline"]["heatmap"]["n_iter"]
    config["top_k"] = config["pipeline"]["restoration"]["top_k"]
    return config