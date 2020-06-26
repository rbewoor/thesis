## References used in the project:
## Webiste: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
## Github: On 05.06.2020, forked from https://github.com/jbrownlee/keras-yolo3 to https://github.com/rbewoor/keras-yolo3

## Goal: Create a YOLOv3 pre-trained model. Present new images to identify the objects. Store the info about which
##       image file has what objects with what score.
##       (:Image{name, dataset}) - HAS{score} -> (:Object{name})
##       dataset property is to track the source of the image:
##              e.g. coco80 for images from COCO dataset running 80 class detector

## This script creates an 80 class detector.

import argparse
import os
import numpy as np
import struct
import cv2

from py2neo import Graph

import json
import sys

## changed np.set_printoptions as per https://github.com/numpy/numpy/issues/12987
## was getting error
## ValueError: threshold must be non-NAN, try sys.maxsize for untruncated representation
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(2**31-1)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='update neo4j graph using saved job output arrays of neo array data')

argparser.add_argument(
    '-sf',
    '--statusfrequency',
    type=int,
    help='show status after how many inserts to neo4j')

argparser.add_argument(
    '-iploc',
    '--inputfilelocationneoarray',
    help='location where the neo4j array are dumped to files')

def update_neo4j_db(_neo_data, _print_status_freq):
    # neo4j access stuff
    neo_uri = r"bolt://localhost:7687"
    auth=(r"neo4j",r"abc")
    stmt1 = r'MERGE (:Image {name: $_in_img_name, dataset: $_in_img_dataset})'
    stmt2 = r'MERGE (:Object {name: $_in_obj_det})'
    stmt3 = r'MATCH (i1:Image{name: $_in_img_name, dataset: $_in_img_dataset}) MATCH (o1:Object{name: $_in_obj_det}) CREATE (i1)-[:HAS{score: $_in_det_score}]->(o1)'
    total_images_info_count = len(_neo_data)
    try:
        graph = Graph(uri="bolt://localhost:7687",auth=("neo4j","abc"))
        for count, each_img_info in enumerate(_neo_data):
            if count % _print_status_freq == 0:
                print(f"\t\tProcessing image {count + 1} of {total_images_info_count}") 
            tx = graph.begin()
            # create Image node if not already existing
            tx.run(stmt1, parameters={"_in_img_name": each_img_info["img"], "_in_img_dataset": each_img_info["datasource"]})
            for each_detection in each_img_info["det"]:
                # create Object node if not already existing
                tx.run(stmt2, parameters={"_in_obj_det": each_detection[0]})
                # create HAS relation between above nodes. Note by now the image and object nodes must exist
                tx.run(stmt3, parameters={"_in_img_name": each_img_info["img"], "_in_img_dataset": each_img_info["datasource"], "_in_obj_det": each_detection[0], "_in_det_score": each_detection[1]})
            tx.commit()
            while not tx.finished():
                pass # tx.finished return True if the commit is complete
    except Exception as error_msg_neo_write:
        print(f"\n\nUnexpected ERROR attempting entry to neo4j.")
        print(f"\nMessage:\n{error_msg_neo_write}")
        print(f"\nFunction call return with RC=1000.\n\n")
        return(1000)
    # return with RC = 0 as successful processing
    return 0

def _main_(args):
    # process command line arguments
    status_freq    = args.statusfrequency      # -sf parameter, after how many images info is processed for neo4j inserts should a status message be shown
    iploc          = args.inputfilelocationneoarray  # -nipt parameter, how many images files to be processed in each task

    # check the status_frequency is valid
    if status_freq < 1:
        print(f"FATAL ERROR: status_frequency argument must be a a non-zero whole number.\nExiting with RC=100")
        exit(100)
    
    # check valid input for the -iploc parameter
    if iploc[-1] != r"/":
        iploc += r"/"
    if not os.path.isdir(iploc):
        print(f"FATAL ERROR: Input for iploc parameter is not an existing directory.\nExiting with RC=100")
        exit(100)
    
    # make array of ALL files in the input directory
    neo_arr_files = [os.path.join(iploc, f) for f in os.listdir(iploc) if os.path.isfile(os.path.join(iploc, f))]
    neo_arr_files_count = len(neo_arr_files)

    ## DEBUGGING
    #for each_file in neo_arr_files:
    #    print(f"{each_file}")
    #exit(0)
    ## DEBUGGING

    # reload each file and use it to update the Neo4j graph
    for each_file in neo_arr_files:
        with open(each_file, "r") as saved_file:
            neo_data = json.load(saved_file)
            print(f"\n\nReloaded array from saved file: {each_file}\nFirst 2 entries of data for Neo4j array:\n{neo_data[:2]}")
            print(f"\nAttempting Neo4j db updates....\n")
            print(f'Status Neo4j db updates: {"SUCCESS" if not update_neo4j_db(neo_data, status_freq) else "PROBLEM"}')
    
    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)