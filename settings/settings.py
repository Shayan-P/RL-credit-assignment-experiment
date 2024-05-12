import os


def cd_mkdir(dir, child, error_if_not_exist=False):
    new_dir = os.path.join(dir, child)
    if not os.path.exists(new_dir):
        if error_if_not_exist:
            raise Exception(f"{dir}/{child} does not exist")
        os.makedirs(new_dir)
    return new_dir


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = cd_mkdir(PROJECT_DIR, "logs")
DATASET_DIR = cd_mkdir(PROJECT_DIR, "datasets")
