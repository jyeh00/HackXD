#!/usr/bin/env python
import sys
import os
import pandas as pd
from extract_training_lables_csv import extract_training_labels_csv, class_text_to_int


def create_label_contents(csv_input_for_labels):
    print ("Opening CSV...")
    examples = pd.read_csv(csv_input_for_labels)
    print ("Opened CSV...")
    unique_label_array = extract_training_labels_csv(examples)
    label_contents = ""
    for lable in unique_label_array:
        print ("Generating Index for lable=="+str(lable))
        index = class_text_to_int(lable, unique_label_array)
        label_contents += "item {\n    id : "+str(index)+"\n    name : '"+str(lable)+"'\n}\n"
    
    return label_contents

def generate_pbtxt_files(file_path, csv_input_for_labels):
    print ("Openening file=="+str(file_path))
    file = open(file_path,'w')
    #contents = "item {\nid : 1\nname : 'mira_robot'\n}\nitem {\nid: 2\nname: 'object'\n}"
    print ("Start create_label_contents...")
    contents = create_label_contents(csv_input_for_labels)
    print ("Done create_label_contents...")
    file.write(contents)
    file.close() 
    print ("Pbtxt Generated..."+str(file_path))
  
if __name__ == "__main__":
    
    file_path = "training/object-detection.pbtxt"
    csv_input_for_labels = "data/train_labels.csv"
    generate_pbtxt_files(file_path, csv_input_for_labels)
