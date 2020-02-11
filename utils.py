import os


def get_log_path(path, log_folder='', log_filename=''):
    if bool(log_folder.strip()):
        log_path = os.path.join(path, log_folder)  # existing sub folder
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if bool(log_filename.strip()):  # existing sub folder
        log_path = os.path.join(log_path, log_filename)

    return log_path
