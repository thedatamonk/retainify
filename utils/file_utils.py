from fastapi import UploadFile

def check_valid_csv(file: UploadFile):
    if file.filename.endswith('csv'):
        return True
    else:
        return False
    # TODO: We can check schema of the csv file
