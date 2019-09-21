import sys
from os import path
from os import listdir
from os.path import isfile, join

def extract_newest_ckpt_name(ckpt_folder_path, output_data_file_path):
    """
    It serach inside the ckpt_folder_path for ckpt files, and saves into the 
    output_data_file_path the one with the biggest number, which is the most recent one.
    """
    files_list = [f for f in listdir(ckpt_folder_path) if isfile(join(ckpt_folder_path, f))]
    print (str(files_list))
    matching = [s for s in files_list if "model.ckpt-" in s]
    print (str(matching))
    meta_files_list = [s for s in matching if ".meta" in s]
    print (str(meta_files_list))
    
    # Get the highest number file model.ckpt-XXX.meta
    MAX_index_file_version = 0
    for meta_file in meta_files_list:
        # filename = model.ckpt-XXX
        filename, file_extension = path.splitext(meta_file)
        # aux1 = XXX.meta
        index_file_version = int(filename.split("-")[1])
        if index_file_version > MAX_index_file_version:
            MAX_index_file_version = index_file_version
    
    
    print ("MAX_INDEX="+str(MAX_index_file_version))
    
    print ("Opening file=="+str(output_data_file_path))
    file = open(output_data_file_path,'w')
    print ("Start create_label_contents...")
    contents = str(MAX_index_file_version)
    print ("Done create_label_contents...")
    file.write(contents)
    file.close() 
    print ("Pbtxt Generated..."+str(output_data_file_path))
    
    return None

if __name__ == "__main__":
    """
    python scripts/extract_newest_ckpt_name.py /home/user/simulation_ws/src/tensorflow_image_automatic_learning/models/research/object_detection/training/ /home/user/simulation_ws/src/tensorflow_image_automatic_learning/newest_ckpt.txt
    """
    ckpt_folder_path= sys.argv[1] 
    output_data_file_path= sys.argv[2]
    extract_newest_ckpt_name(ckpt_folder_path, output_data_file_path)
