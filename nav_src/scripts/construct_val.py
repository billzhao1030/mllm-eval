'''
Construct the validation set for zero-shot evaluation.
extract the first trajectory for each scene in MP3D.
'''

import os
import json
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))

            with open(filepath) as f:
                new_data = json.load(f)

        data += new_data
    return data

def construct_instrs_action_plan(anno_dir, dataset, splits):
    data = []
    scan_dict = {}
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        if item['scan'] not in scan_dict:
            scan_dict[item['scan']] = 1
        elif scan_dict[item['scan']] >= 1:
            continue
        else:
            scan_dict[item['scan']] += 1
        data = data + [item]
    return data

def construct_instrs(anno_dir, dataset, splits):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        # Split multiple instructions into separate entries 
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

# def main():
#     splits = ['train', 'val_seen', 'val_unseen']
#     anno_dir = './datasets/R2R/annotations'
#     dataset = 'R2R'
#     data = construct_instrs(anno_dir, dataset, splits)
#     with open('./datasets/R2R/annotations/R2R_val_72.json', 'w') as f:
#         json.dump(data, f)

def main():
    splits = ['val_unseen']
    anno_dir = './datasets/R2R/annotations'
    dataset = 'R2R'
    data = construct_instrs(anno_dir, dataset, splits)
    with open('./datasets/R2R/annotations/R2R_val_unseen_instr.json', 'w') as f:
        json.dump(data, f)

def filter_longest_action_plans(path):
    samples = json.load(open(path, "r"))
    path_id_to_sample = {}

    for sample in samples:
        path_id = sample["path_id"]
        action_plan_length = len(sample["action_plan"])

        if (path_id not in path_id_to_sample or action_plan_length > len(path_id_to_sample[path_id]["action_plan"])) and 'wait' not in sample["action_plan"]:
            path_id_to_sample[path_id] = sample

    return list(path_id_to_sample.values())

def filter_instructions(path):
    '''
    Filter the instructions with the same instructions id. e.g. 121_0, 122_0, 123_0 
    '''
    samples = json.load(open(path, "r"))
    sample_list0 = []
    sample_list1 = []
    sample_list2 = []

    for sample in samples:
        instr_id = sample["instr_id"]
        instr_id, idx = instr_id.split("_")
        if idx == "0":
            sample_list0.append(sample)
        elif idx == "1":
            sample_list1.append(sample)
        elif idx == "2":
            sample_list2.append(sample)
    
    return sample_list0, sample_list1, sample_list2

def split_val_unseen(path):
    '''
    split the val_unseen into 5 parts
    '''
    samples = json.load(open(path, "r"))
    sample_list0 = samples[:470]
    sample_list1 = samples[470:940]
    sample_list2 = samples[940:1410]
    sample_list3 = samples[1410:1880]
    sample_list4 = samples[1880:]

    return sample_list0, sample_list1, sample_list2, sample_list3, sample_list4

if __name__ == "__main__":
    # main()
    # filtered_data = filter_longest_action_plans("./datasets/R2R/annotations/R2R_val_72_action_plan.json")
    # with open("./datasets/R2R/annotations/R2R_val_72_action_plan_filtered.json", "w") as f:
    #     json.dump(filtered_data, f)

    # filtered_data_0, fileter_data_1, fileter_data_2 = filter_instructions("./datasets/R2R/annotations/R2R_val_72_action_plan.json")
    # data_list = [filtered_data_0, fileter_data_1, fileter_data_2]
    # for i in range(3):
    #     assert len(data_list[i]) == 72, "The number of samples in the {}th split is not 72.".format(i)
    #     with open("./datasets/R2R/annotations/R2R_val_72_instr_{}.json".format(i), "w") as f:
    #         json.dump(data_list[i], f, indent=4)

    filtered_data_0, fileter_data_1, fileter_data_2, fileter_data_3, fileter_data_4 = split_val_unseen("./datasets/R2R/annotations/R2R_val_unseen_instr.json")
    data_list = [filtered_data_0, fileter_data_1, fileter_data_2, fileter_data_3, fileter_data_4]
    length = 0
    for i in range(5):
        length += len(data_list[i])
        if i == 4:
            assert length == 2349, "The number of samples in the total split is not 2349."
        with open("./datasets/R2R/annotations/R2R_val_unseen_instr_{}.json".format(i), "w") as f:
            json.dump(data_list[i], f, indent=4)