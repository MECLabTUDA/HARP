import json
import os
import argparse
import harp.core.praser as Praser

def manage_path(dir):
    base_path = os.path.dirname(os.path.dirname(__file__))
    return  os.path.normpath(os.path.join(base_path, dir))

def load_config(file_path):
    json_str = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str)
    return config

def parse(config_path):
    config = load_config(config_path)
    config["input_folder_path"] = config["pipeline"]["input_folder_path"]
    config["result_folder_path"] = config["pipeline"]["result_folder_path"]
    config["batch_size"] = config["pipeline"]["batch_size"]
    config["gpu_id"] = config["pipeline"]["gpu_id"]
    config["image_size"] = config["pipeline"]["image_size"]
    config["save_images"] = config["pipeline"]["save_restored_image_mask"]
    
    config["anomalib_config_path"] = manage_path(config["pipeline"]["anomalib"]["config_path"])
    config["anomalib_model_path"] = config["pipeline"]["anomalib"]["model_path"]
    config["anomalib_prediction_threshold"] = config["pipeline"]["anomalib"]["prediction_threshold"]
    
    config["sam_model_type"] = config["pipeline"]["sam"]["model_type"]
    config["sam_checkpoint"] = config["pipeline"]["sam"]["sam_checkpoint"]

    config["n_noising_step"] = config["pipeline"]["heatmap"]["n_noising_step"]
    config["n_iter"] = config["pipeline"]["heatmap"]["n_iter"]
    
    config["restoration_config_path"] = manage_path(config["pipeline"]["restoration_model"]["config_path"])
    config["restoration_model_path"] = config["pipeline"]["restoration_model"]["model_path"]
    config["restoration_top_k"] = config["pipeline"]["restoration_model"]["top_k"]

    args = argparse.Namespace(config=config["restoration_config_path"], 
                              phase="test",
                              batch=config["batch_size"],
                              gpu_ids=str(config["gpu_id"]),
                              debug=False)
    opt = Praser.parse(args)
    config["config_restoration"] = opt

    return config


