import json

def generate_dataloader(file, batch_size=1024):
    with open(file, "r") as file:
        data = json.load(file)
    start_id = 0
    end_id = batch_size
    is_end = (len(data) <= start_id)
    while not is_end:
        chunk = data[start_id : end_id].copy()
        start_id += batch_size
        end_id += batch_size
        end_id = end_id if end_id <= len(data) else len(data)
        is_end = (len(data) <= start_id)
        yield chunk


