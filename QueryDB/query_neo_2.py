## -----------------------------------------------------------------------------
## Goal: Query Neo4j database to find images that contain the objects of interest. The objects
##        of interest are read from a file specified in a command line argument.
## -----------------------------------------------------------------------------
##         Neo4j database schema:
##             (:Image{name, dataset}) - HAS{score} -> (:Object{name})
##             Image Nodes:
##             - name property is the name of the image file  
##             - dataset property is to track the source of the image:
##               e.g. "coco_val_2017" for images from COCO validation 2017 dataset
##               e.g. "coco_test_2017" for images from COCO test 2017 dataset
##               e.g. "flickr30" for images from Flickr 30,000 images dataset
##             Object Nodes:
##             - name property is the label of the object from the object detector.
##             - datasource property used to keep track of which dataset this image belongs to
##             HAS relationship:
##             - score property is confidence score of the object detector. Value can theoretically be from 0.00 to 100.00
## -----------------------------------------------------------------------------
##         The various objects that can be searched for currently are as follows:
## 
##         labels = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', \
##                   'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', \
##                   'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', \
##                   'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', \
##                   'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', \
##                   'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', \
##                   'orange', 'oven', 'parking meter', 'person', 'pizza', 'pottedplant', \
##                   'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', \
##                   'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', \
##                   'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', \
##                   'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -objarrfile : location of file specifying the objects to search for during database query.
##                 The file contains a list of lists.
##                 Outer list must contain either 2 or 3 elements.
##                 Each inner list must have exactly 1/ 2/ 3 elements. Each of these elements must be a string.
##                 Below are examples of two files. E.g. of file contents.
##                      Contents of file 1:
##                          [ ["clock", "book"], ["person", "bird", "clock"] ]
##                      Contents of file 2:
##                          [ ["dog", "cat"], ["person", "book"], ["person", "car"] ]
## -----------------------------------------------------------------------------
## Usage example:
##    python3 query_neo_2.py -objarrfile "/home/rohit/PyWDUbuntu/thesis/queryDb/query_db_input_test_dblquote.txt"
## -----------------------------------------------------------------------------

import argparse
import os
import sys
import json

from py2neo import Graph

argparser = argparse.ArgumentParser(
    description='query Neo4j graph db to return images based on query objects')
 
argparser.add_argument(
    '-objarrfile',
    '--inputobjectoarrayfile',
    help='file containing array of three arrays of string elements corresponding to the key element i.e. objects. Each inner array is for one input sentence spoken.')

def check_input_data(_key_elements_list):
    '''
    Perform santiy check on the input data.
    1) Check it is a list of lists.
    2) Outer list must contain exactly 2 or 3 elements.
    3) Each inner list to consist of 1 or 2 or 3 elements.
    4) All inner list elements to be strings.
    VARIABLES PASSED:
          1) list containing the objects to find within individual image
    RETURNS:
          1) return code = 0 means all ok, non-zero means problem
            value       error situation
            100         main data structure is not a list
            105         outer list did not contain exacty 2/3 elements
            110         some inner element is not a list
            115         some inner list did not contain exactly 1/2/3 elements
            120         some element of the inner list is not a string
    '''
    if type(_key_elements_list) != list:
        print(f"\n\nERROR 100: Expected a list of lists as the input data. Outer data is not a list. Please check the input and try again.\n")
        return 100
    elif len(_key_elements_list) not in [2, 3]:
        print(f"\n\nERROR 105: Expected a list with exactly two/ three elements. Please check the input and try again.\n")
        return 105
    for inner_list in _key_elements_list:
        if type(inner_list) != list:
            print(f"\n\nERROR 110: Expected a list of lists as the input data. Inner data is not a list. Please check the input and try again.\n")
            return 110
        elif len(inner_list) not in [1, 2, 3]:
            print(f"\n\nERROR 115: Some inner list did not contain exactly one/two/three elements. Please check the input and try again.\n")
            return 115
        for object_string in inner_list:
            if type(object_string) != str:
                print(f"\n\nERROR 120: Expected a list of lists as the input data, with inner list consisting of strings. Please check the input and try again.\n")
                return 120
    return 0 ## all ok

def query_neo4j_db(_each_inner_list, _in_limit):
    '''
    Queries database and returns the distinct results.
    VARIABLES PASSED:
          1) list containing the objects to find within individual image
          2) limit value for number of results from neo4j
    RETURNS:
          tuple of (return code, results as list of dictionarary items)
    Return code = 0 means all ok
            value       error situation
            100         unable to connect to database
            1000        unexpected situation - should not occur ever - length of the inner list passed was not one/ two/ three
            1500        problem querying the neo4j database
    '''
    ## set result as None by default
    result = None
    
    ## establish connection to Neo4j
    try:
        graph = Graph(uri="bolt://localhost:7687",auth=("neo4j","abc"))
    except Exception as error_msg_neo_connect:
        print(f"\n\nUnexpected ERROR connecting to neo4j.")
        print(f"\nMessage:\n{error_msg_neo_connect}")
        print(f"\nFunction call return with RC=100.\n\n")
        return (100, result)
    
    ## build the query and execute
    stmt1_three_objects = r'MATCH (o1:Object)--(i:Image)--(o2:Object)--(i)--(o3:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 AND o3.name = $in_onj3 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt2_two_objects = r'MATCH (o1:Object)--(i:Image)--(o2:Object) ' + \
        r'WHERE o1.name = $in_obj1 AND o2.name = $in_obj2 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    stmt3_one_objects = r'MATCH (o1:Object)--(i:Image) ' + \
        r'WHERE o1.name = $in_obj1 ' + \
        r'RETURN DISTINCT i.name as Image, i.dataset as Source ' + \
        r'LIMIT $in_limit'
    try:
        tx = graph.begin()
        if len(_each_inner_list) == 3:
            in_obj1, in_obj2, in_onj3 = _each_inner_list
            result = tx.run(stmt1_three_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_onj3': in_onj3, 'in_limit': _in_limit}).data()
        elif len(_each_inner_list) == 2:
            in_obj1, in_obj2 = _each_inner_list
            result = tx.run(stmt2_two_objects, parameters={'in_obj1': in_obj1, 'in_obj2': in_obj2, 'in_limit': _in_limit}).data()
        elif len(_each_inner_list) == 1: # MUST be true
            in_obj1 = _each_inner_list[0]
            result = tx.run(stmt3_one_objects, parameters={'in_obj1': in_obj1, 'in_limit': _in_limit}).data()
        else:
            print(f"\n\nUnexpected length of the key elements array. Result set to None.")
            print(f"\nFunction call return with RC=1000.\n\n")
            return (1000, result)
        #tx.commit()
        #while not tx.finished():
        #    pass # tx.finished return True if the commit is complete
    except Exception as error_msg_neo_read:
        print(f"\n\nUnexpected ERROR querying neo4j.")
        print(f"\nMessage:\n{error_msg_neo_read}")
        print(f"\nFunction call return with RC=1500.\n\n")
        return (1500, result)
    ## return tuple of return code and the results. RC = 0 means no errors.
    return (0, result)

def query_neo4j_functionality(args):
    ## process command line arguments
    objarrfile = args.inputobjectoarrayfile ## -objarrfile parameter - contains location of file with contents like e.g. [ ['person', 'dog'], ['chair'], ['truck', 'train'] ]
    if not os.path.isfile(objarrfile):
        print(f"\nInput parameter is not a file. Exiting with return code 200.")
        exit(200)
    
    ## open the file and load the array
    key_elements_list = None
    try:
        with open(objarrfile, 'r') as f:
            key_elements_list = json.load(f)
    except Exception as e:
        print(f"\nProblem loading data from the file. Exiting with return code 250.")
        print(f"Error message: {e}")
        exit(250)
    
    ## perform sanity checks on the file contents
    ### DEBUG -start
    print(f"\n\ntype key_elements_list = {type(key_elements_list)}\nkey_elements_list =\n{key_elements_list}")
    ### DEBUG - end
    if check_input_data(key_elements_list):
        print(f"\nProblem with the input data. Exiting with return code 300.")
        exit(300)
    
    query_results_arr = []
    ## query the database with the input data
    query_result_limit = 10
    for each_inner_list in key_elements_list:
        query_rc, query_result = query_neo4j_db(each_inner_list, query_result_limit)
        if query_rc != 0:
            print(f"\n\nProblem retrieving data from Neo4j; query_rc = {query_rc}.\nExiting with return code 500.")
            exit(500)
        else:
            print(f"\nQuery result for: {each_inner_list}\n{query_result}\n")
            query_results_arr.append(query_result)
    
    ## show results
    print(f"\n\nFinal results=\n{query_results_arr}")

    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    query_neo4j_functionality(args)
