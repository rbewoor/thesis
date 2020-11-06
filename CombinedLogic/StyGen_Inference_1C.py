## -----------------------------------------------------------------------------
###             FUNCTIONAL programming approach
### Goal: Process the output of the edited or not Image Caption stage output.
###       Use the captions to generate the short story using the Story Generator.
## -----------------------------------------------------------------------------
##  STAGE 7) Story Generation using fine-tuned GPT-2 model (355M).
##        Use the captions (original or edited as the case may be) to generate short story.
##        Logic:
##           a) Make combinations of seed sentences by taking on sentence for each query.
##           b) Use the combination seed sentences to create on story.
##           c) Create multiple stories by using different combinations of seed sentences.
##           d) Score the stories for grammar, etc.
##       Input:
##        a) Intermediate data from previous stage:
##           File containing the data structure as input.
##       Outputs:
##        a) Text file containing the generated texts.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -micYorN          : Wwitch to bypass/include mic input stage execution.
## 2) -sw_here          : If mic input stage executed, then the newly created wav files ot be saved in this directory.
##                        But if mic input stage is bypassed, then irrelevant.
## 3) -file4STT         : This is the file used by STT stage for inference. Expects each line to have the path for one file. Therefore, three lines in file expected.
##                        If mic input stage is executed, then this file will be created (with entries of absolute paths of the newly created wav files).
##                        But if mic input stage is bypassed, then this file MUST already exist with required three entries with the three wav files existing at those locations specified.
## 4) -op_res_file      : File to save the results to
## 5) -opfileallposinfo : Location where to write a file with parts-of-speech (POS) info during Identify key elements stage.
## 6) -logfileloc       : Location to write the log file.
## -----------------------------------------------------------------------------
## Usage example:
## 
## python3 StyGen_Inference_1C.py -file4stygen "/home/rohit/PyWDUbuntu/thesis/combined_execution/ImgCapAfterGui/op_img_cap_after_gui.txt" -ckptdir "/home/rohit/PyWDUbuntu/thesis/StyGen/checkpoint/" -run "Run2_File11_2_checkpoint_run2" -logfileloc "./LOG_StyGen_Inference_1C.LOG"
## -----------------------------------------------------------------------------

## import necessary packages

##   imports for common or generic use packages
#from __future__ import print_function
import argparse
import os
import json
import time
import datetime
import subprocess
import copy
#import string
#import pickle
import numpy as np

import logging

##   imports for gui
#import tkinter as tk
#from functools import partial
#from PIL import ImageTk, Image
#from keras.models import Model as keras_Model, load_model as keras_load_model
#import struct
#import cv2

##   imports for image captioning inference - no gui stage
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

##   imports for story generation
import gpt_2_simple as gpt2
from gpt_2_simple.src import model, sample, encoder, memory_saving_gradients
from gpt_2_simple.src.load_dataset import load_dataset, Sampler
from gpt_2_simple.src.accumulate import AccumulatingOptimizer

## command line arguments
argparser = argparse.ArgumentParser(
    description='parameters to run this program')

argparser.add_argument(
    '-file4stygen',
    '--ipfile_for_sty_gen',
    help='location to pick results for story generator logic to process')

argparser.add_argument(
    '-op_res_file',
    '--opfile_for_sty_gen_results',
    help='location to store the data structure of images and their stories')

argparser.add_argument(
    '-ckptdir',
    '--checkpoint_directory',
    help='location of GPT-2 checkpoint directory')

argparser.add_argument(
    '-run',
    '--run_name',
    help='folder name of run in the checkpoint directory')

argparser.add_argument(
    '-logfileloc',
    '--oplogfilelocation',
    help='location for output file for logging')

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -------------------     STORY GENERATOR STARTS                  STORY GENERATOR STARTS   -------------------
###############################################################################################################
def my_load_gpt2(_sess,
              _checkpoint='latest',
              _run_name="run1",
              _checkpoint_dir="checkpoint",
              _model_name=None,
              _model_dir='models',
              _multi_gpu=False):
    """
    Loads the model checkpoint or existing model into a TensorFlow session for repeated predictions.
    """

    if _model_name:
        checkpoint_path = os.path.join(_model_dir, _model_name)
    else:
        checkpoint_path = os.path.join(_checkpoint_dir, _run_name)
    print(f"\ncheckpoint_path = {checkpoint_path}\n")

    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    context = tf.compat.v1.placeholder(tf.int32, [1, None])

    gpus = []
    if _multi_gpu:
        gpus = get_available_gpus()

    output = model.model(hparams=hparams, X=context, gpus=gpus)

    if _checkpoint=='latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
    else:
        ckpt = os.path.join(checkpoint_path,_checkpoint)

    saver = tf.compat.v1.train.Saver(allow_empty=True)
    _sess.run(tf.compat.v1.global_variables_initializer())

    if _model_name:
        print(f"\nLoading pretrained model :: {ckpt}\n")
    else:
        print(f"\nLoading checkpoint :: {ckpt}\n")
    saver.restore(_sess, ckpt)

def reset_session(_sess, threads=-1, server=None):
    """Resets the current TensorFlow session, to clear memory
    or load another model.
    """

    tf.compat.v1.reset_default_graph()
    _sess.close()

def stg_gen_do_one_inference(_seed_string, _ckptdir, _runName, _DEBUG_SWITCH):
    ## Parameters to generate the story
    ## NOTE: nsamples % batch_size == 0
    SG_LENGTH = 75
    SG_TEMPERATURE = 0.85
    SG_NSAMPLES = 2
    SG_BATCH_SIZE = 2
    #RETURN_LIST = True ## hardcoded as True during function call

    ## RUN_NAME must match a folder in the checkpoint directory
    #RUN_NAME = r"Run2_File11_2_checkpoint_run2"

    ## check samples and batch size values are compatible
    assert (SG_NSAMPLES % SG_BATCH_SIZE == 0) , f"Values for NSAMPLES and BATCH_SIZE incompatible. " \
            f"NSAMPLES={SG_NSAMPLES} % SG_BATCH_SIZE={SG_BATCH_SIZE} != 0"
    
    ## set up the pre-trained GPT-2 model from checkpoint directory
    
    ## copy checkpoint directory from gdrive - not required as saved it locally
    ##gpt2.copy_checkpoint_from_gdrive(run_name='run1')

    sess = gpt2.start_tf_sess()
    myStr = f"\nTF session started\n"
    print_and_log(myStr, "debug")
    myStr = None
    
    my_load_gpt2(
        _sess = sess,
        _checkpoint='latest',
        _run_name=_runName,
        _checkpoint_dir=_ckptdir,
        _model_name=None,
        _model_dir='models',
        _multi_gpu=False)
    
    myStr = f"\n\n\tGenerating the story using 'prefix' =\n{_seed_string}\n"
    print_and_log(myStr, "info")
    myStr = None

    story_as_list = gpt2.generate(
        sess,
        length=SG_LENGTH,
        temperature=SG_TEMPERATURE,
        prefix=_seed_string,
        nsamples=SG_NSAMPLES,
        batch_size=SG_BATCH_SIZE,
        checkpoint_dir=_ckptdir,
        return_as_list=True
        )
    
    reset_session(sess)

    ## for each story, split on new line and rejoin with space
    for i, each_story in enumerate(story_as_list):
        split_sents = each_story.split('\n')
        story_as_list[i] = ' '.join(split_sents)

    myStr = f"\n\n\tGENERATED STORY after removing newline characters =\n{story_as_list}\n\n"
    print_and_log(myStr, "info")
    myStr = None
    return story_as_list

def sty_gen_create_triplets(_prev_stage_info, _DEBUG_SWITCH):
    list1, list2, list3 = [ list(range( len(q_info['selected_images']) )) for q_info in _prev_stage_info]
    myStr = '\n'.join([
        f"\nLists for triplets BEFORE inserting -1:",
        f"list1 = \n{list1}",
        f"list2 = \n{list2}",
        f"list3 = \n{list3}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    print(f"list1 = {list1}\nlist2 = {list2}\nlist3 = {list3}\n")
    ## insert -1 as dummy value into empty lists. Prevents the for loop later from not executing
    if not list1: list1.append(-1)
    if not list2: list2.append(-1)
    if not list3: list3.append(-1)
    myStr = '\n'.join([
        f"\nLists for triplets AFTER inserting -1:",
        f"list1 = \n{list1}",
        f"list2 = \n{list2}",
        f"list3 = \n{list3}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    results = list()
    for v1 in list1:
        for v2 in list2:
            for v3 in list3:
                results.append((v1, v2, v3))
    return results

def sty_gen_inference_functionality(_file4stygen, _ckptdir, _runName, _opfile, _DEBUG_SWITCH):
    ## load the data of previous stage from the saved file
    try:
        with  open(_file4stygen, "r") as f:
            gui_img_cap_module_results = json.load(f)
        myStr = "\n".join([
            f"\nSuccessfully reloaded data from previous stage",
            f"gui_img_cap_module_results =\n{gui_img_cap_module_results}\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
    except Exception as stg_gen_ipfile_reload_problem_msg:
        myStr = '\n'.join([
            f"\nFATAL ERROR: Problem reloading data from previous stage from file =\n{_file4stygen}",
            f"Error message =\n{stg_gen_ipfile_reload_problem_msg}",
            f"Exiting with Return Code = 200\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(200)
    
    """
    ##### temp code - start - no empty list
    gui_img_cap_module_results = \
    [
        {
            'key_elements': ['bicycle', 'person'],
            'selected_images': [
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000155758.jpg', 'man riding bike in a park']
                ]
                },
        {
            'key_elements': ['tvmonitor', 'person'],
            'selected_images': [
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000287296.jpg', 'young girl standing in front of tv playing video game'],
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000175337.jpg', 'man walking past a tv in his house']
                ]
                },
        {
            'key_elements': ['handbag'],
            'selected_images': [
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000117100.jpg', 'two suitcases are in the boot of a car while a woman holds a hand bag in her hand'],
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000313060.jpg', 'woman is holding her hang bag while speaking on her cell phone']
                ]
                }
    ]
    ##### temp code - end
    """
    #"""
    ##### temp code - start - with empty list
    gui_img_cap_module_results = \
    [
        {
            'key_elements': ['bicycle', 'person'],
            'selected_images': []
                },
        {
            'key_elements': ['tvmonitor', 'person'],
            'selected_images': [
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000287296.jpg', 'young girl standing in front of tv playing video game'],
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000175337.jpg', 'man walking past a tv in his house']
                ]
                },
        {
            'key_elements': ['handbag'],
            'selected_images': [
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000117100.jpg', 'two suitcases are in the boot of a car while a woman holds a hand bag in her hand'],
                ['/media/rohit/DATA/EverythingD/01SRHBDBA_Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/000000313060.jpg', 'woman is holding her hang bag while speaking on her cell phone']
                ]
                }
    ]
    ##### temp code - end
    #"""

    ## make copy for results of this stage - add empty list to store generated stories
    sty_gen_inference_module_results = copy.deepcopy(gui_img_cap_module_results)

    ## create sets of all combinations possible containing exactly one image from each of the queries
    ## Note: -1 is dummy to indicate no image for that query
    triplets = sty_gen_create_triplets(gui_img_cap_module_results, _DEBUG_SWITCH)
    myStr = f"\nTriplets BEFORE any -1 insertions =\n{triplets}\n"
    print_and_log(myStr, "error")
    myStr = None

    triplet_seeds_for_styGen = []
    triplet_imgs_for_stgGen = []
    for triplet in triplets:
        trip_seeds = []
        trip_imgs  = []
        q1idx, q2idx, q3idx = triplet
        if q1idx == -1:
            trip_seeds.append(None)
            trip_imgs.append(None)
        else:
            trip_seeds.append(gui_img_cap_module_results[0]['selected_images'][q1idx][1].capitalize() + r'.' )# + r' ')
            trip_imgs.append(gui_img_cap_module_results[0]['selected_images'][q1idx][0])
        if q2idx == -1:
            trip_seeds.append(None)
            trip_imgs.append(None)
        else:
            trip_seeds.append(gui_img_cap_module_results[1]['selected_images'][q2idx][1].capitalize() + r'.' )# + r' ')
            trip_imgs.append(gui_img_cap_module_results[1]['selected_images'][q2idx][0])
        if q3idx == -1:
            trip_seeds.append(None)
            trip_imgs.append(None)
        else:
            trip_seeds.append(gui_img_cap_module_results[2]['selected_images'][q3idx][1].capitalize() + r'.' )# + r' ')
            trip_imgs.append(gui_img_cap_module_results[2]['selected_images'][q3idx][0])
        triplet_seeds_for_styGen.append(trip_seeds)
        triplet_imgs_for_stgGen.append(trip_imgs)
    #seed_for_styGen = [
    #    r"a person is riding on a bicycle on the side of the road",
    #    r"dogs are playing in the park",
    #    r"a person is speaking on a phone"]
    
    myStr = f"Triplets AFTER any -1 insertions =\n{triplet_seeds_for_styGen}\nAll imgs =\n{triplet_imgs_for_stgGen}\nTotal triples = {len(triplet_seeds_for_styGen)}\n"
    print_and_log(myStr, "info")
    myStr = None
    #print(f"All seeds =\n{triplet_seeds_for_styGen}\nAll imgs =\n{triplet_imgs_for_stgGen}\n")
    #for idx, (triplet, seeds, imgs) in enumerate(zip(triplets, triplet_seeds_for_styGen, triplet_imgs_for_stgGen)):
    #    print(f"\nentry {idx}\t\ttriplet = {triplet}\nseeds = {seeds}\nimgs = {imgs}")
    
    ## make copy for results of this stage
    sty_gen_inference_module_results = copy.deepcopy(gui_img_cap_module_results)
    ## add dict entries of empty lists to store generated stories - later tuples of story and associated image/s will be added to this
    ##     make entry to hold the individual sentence stories
    sty_gen_inference_module_results.append({'individual_sent_stories': []})
    ##     make entry to hold the combined sentences stories
    sty_gen_inference_module_results.append({'combined_sent_stories': []})
    
    ## actually generate the stories from model
    ## 1) individual sentences stories
    for trip_seeds, trip_imgs in zip(triplet_seeds_for_styGen, triplet_imgs_for_stgGen):
        for seed, img in zip(trip_seeds, trip_imgs):
            if seed is not None:
                story_texts = stg_gen_do_one_inference(seed, _ckptdir, _runName, _DEBUG_SWITCH)
                for each_story in story_texts:
                    sty_gen_inference_module_results[3]['individual_sent_stories'].append((img, each_story))
    ## 2) combined sentences stories
    for trip_seeds, trip_imgs in zip(triplet_seeds_for_styGen, triplet_imgs_for_stgGen):
        comb_seed = ' '.join(seed for seed in trip_seeds if seed is not None)
        story_texts = stg_gen_do_one_inference(comb_seed, _ckptdir, _runName, _DEBUG_SWITCH)
        for each_story in story_texts:
            sty_gen_inference_module_results[4]['combined_sent_stories'].append(([img for img in trip_imgs if img is not None], each_story))
    
    ## save results data structure to output file
    try:
        with open(_opfile, "w") as f:
            json.dump(sty_gen_inference_module_results, f)
        myStr = f"\nSaved results data structure to file here: {_opfile}\n"
        print_and_log(myStr, "info")
        myStr = None
    except Exception as opfile_save_results_problem:
        myStr = '\n'.join([
            f"\nFATAL ERROR: Problem saving results data structure to file here: {_opfile}",
            f"Error message =\n{opfile_save_results_problem}",
            f"Exiting with Return Code = 250\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(250)

    return (0, None, sty_gen_inference_module_results)

###############################################################################################################
## ---------------------     STORY GENERATOR ENDS                  STORY GENERATOR ENDS   ---------------------
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ----------------------  CONTROL LOGIC STARTS                    CONTROL LOGIC STARTS ---------------------- 
###############################################################################################################
def print_and_log(_in_fstring_msg, _log_level):
    print(f"LOG_LEVEL {_log_level.upper()} :: {_in_fstring_msg}")
    if _log_level == "debug":
        logging.debug(f"{_in_fstring_msg}")
    elif _log_level == "warning":
        logging.warning(f"{_in_fstring_msg}")
    elif _log_level == "info":
        logging.info(f"{_in_fstring_msg}")
    elif _log_level == "error":
        logging.error(f"{_in_fstring_msg}")
    elif _log_level == "critical":
        logging.critical(f"{_in_fstring_msg}")
    else:
        print(f"\n\n\nFATAL ERROR - wrong parameters passed to print_and_log function\n\n\nExiting with RC=9000\n")
        exit(9000)
    return

def check_command_line_args(_args):
    """
    Following checks performed:
    1) The input file for story generator  MUST exist.
    2) The directory for checkpoint folder MUST exist.
    3) The run name MUST be a directory in the checkpoint folder.
    4) The directory for the output results file MUST exist.
    """
    ## extract command line args and check them
    file4stygen       = args.ipfile_for_sty_gen          ## -file4stygen           parameter
    opfile            = args.opfile_for_sty_gen_results  ## -op_res_file           parameter
    ckptdir           = args.checkpoint_directory        ## -ckptdir               parameter
    runName           = args.run_name                    ## -run                   parameter
    logfileloc        = args.oplogfilelocation           ## -oplogfilelocation     parameter

    if not os.path.isfile(file4stygen):
        return 10, f'Error in CLA parameter "-file4stygen" : Input file does not exist.', None
    
    if ckptdir[-1] != r'/':
        ckptdir += r'/'
    if not os.path.isdir(ckptdir):
        return 15, f'Error in CLA parameter "-ckptdir" : Checkpoint folder does not exist.', None
    
    runDir = ckptdir + runName
    if runDir[-1] != r'/':
        runDir += r'/'
    if not os.path.isdir(runDir):
        cla_err_msg = f'Error in CLA parameter/s "-ckptdir and/or -run" : The run paramter must be a folder in the chekcpoint directory' + \
            f"This is not an existing directory :: {runDir}"
        return 20, cla_err_msg, None
    
    if not os.path.isdir( r'/'.join(opfile.split(r'/')[:-1]) + r'/' ):
        return 25, f'Error in CLA parameter "-op_res_file" : Expected an existing folder in the path specified before the filename', None
    
    return 0, None, [file4stygen, ckptdir, runName, opfile, logfileloc]

def execute_control_logic(args):
    
    DEBUG_SWITCH = False
    
    check_cla_rc, check_cla_msg, returned_check_cla_returned_list = check_command_line_args(args)
    if check_cla_rc != 0:
        print(f"\n\nFATAL ERROR: Failed processing command line arguments with RC = {check_cla_rc}.\nMessage = {check_cla_msg}\n")
        print(f"\nExiting with Return Code = 10\n")
        exit(10)

    ## show CLA being used
    print(f"\nRunning with command line args as follows:")    
    cla_names = ('-file4stygen', '-ckptdir', '-run', '-op_res_file', '-logfileloc')
    for idx, (cla_name, cla_value) in enumerate(zip(cla_names, returned_check_cla_returned_list)):
        print(f"""{idx+1}) {cla_name} = {''.join([ '"', cla_value, '"' ])}""")
    
    file4stygen, ckptdir, runName, opfile, logfileloc = returned_check_cla_returned_list
    del check_cla_rc, check_cla_msg, returned_check_cla_returned_list

    ## setup logging file -   levels are DEBUG , INFO , WARNING , ERROR , CRITICAL
    logging.basicConfig(level=logging.INFO, filename=logfileloc,                               \
        filemode='w', format='LOG_LEVEL %(levelname)s : %(asctime)s :: %(message)s')
    
    #######################################################################
    ## STAGE 7 :: STORY GENERATION LOGIC
    #######################################################################
    ## get stories using saved story gen gpt-2 model
    myStr = "\n".join([
        f"\n\n-------------------------------------------------------------------------------",
        f"-------------------------------------------------------------------------------",
        f"    STAGE 7)  STARTING EXECUTION OF STORY GENERATION                           ",
        f"-------------------------------------------------------------------------------",
        f"-------------------------------------------------------------------------------\n\n"
        ])
    print_and_log(myStr, "info")
    myStr = None

    sty_gen_inference_logic_RC, sty_gen_inference_logic_msg, sty_gen_inference_module_results = sty_gen_inference_functionality(file4stygen, ckptdir, runName, opfile, DEBUG_SWITCH)
    myStr = "\n".join([
        f"\nAfter STORY GENERATION logic execution:",
        f"sty_gen_inference_logic_RC = {sty_gen_inference_logic_RC}",
        f"sty_gen_inference_logic_msg =\n{sty_gen_inference_logic_msg}",
        f"sty_gen_inference_module_results =\n{sty_gen_inference_module_results}\n"
        ])
    print_and_log(myStr, "info")
    myStr = None

    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)

    myStr = f"\n\n\nNormal exit from program.\n"
    print_and_log(myStr, "info")
    myStr = None

    exit(0)
