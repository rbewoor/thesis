## -----------------------------------------------------------------------------
###             FUNCTIONAL programming approach
### Goal: Do STT processing on the input wav files, then do NLP processing to extract the keywords,
###          then query the Neo4j database to return images containing the objects of interest. Per query,
###          limiting results to maximum 20 images. These are the candidate images.
###          From this set of candidate images, allow user to inspect the images via a graphical user
###          interface and Deselect any images. Per Query, a maximum of 5 images can be Selected from the
###          set of up to 20 images available. Thus reduce the total images from up to 60, to maximum 15.
###          User may Deselect ALL the images from the results of a query, if unhappy with all the images. 
## 
##  1) STT part:
##     Use Deepspeech model to run inference on the input wav files.
##     Outputs:
##        a) Intermediate data:
##           for use by the identify keywords logic
## 
##  2) Identify keywords
##        Process the input data which is the speech to text transcriptions.
##        Using Spacy, remove stop-words, perform part of speech tagging.
##        Write all POS info for these non stopwords to a file (word types as noun or verb).
##        But for the candidate keywords pick up only the nouns.
##        Using a GUI present the candidate keywords to user for Deselecting words.
##             Ensure only valid number of keywords are Selected per input sentence.
##        Pass the final data structure of Selected words for processing by database query module.
##     Outputs:
##        a) One text file:
##           For ALL words that are not Stopwords, of type Noun or Verb, their 
##               comprehensive nlp info from Spacy.
##        b) Intermediate data:
##           The data strucutre required for neo4j query logic.
##           Depending on the value of the boolean variable POS_INTEREST_INCLUDE_VERBS_SWITCH
##              either only nouns, or both nouns and verbs, will be retained. Then the user
##              drops any words if required.
## 
##  3) Query Neo4j for images matching the keywords
##     Query Neo4j database to find images that contain the objects of interest.
## 
##     Neo4j database schema:
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
## 
##      Currently, only these objects are in the database:
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
## 
##  4) Graphical user interaction based Deselection of candidate images obtained from Neo4j query stage.
##        Allow user to Enlarge an image, perform object detection inference and finally DESELECT images.
##              This is because some images may have incorrectly found the objects of interest.
##              Each Neo4j query can return up to 20 images. But the maximum number of images per query to
##                   pass to the next stage (Auto caption block) is limited to 5 per query. So allow culling.
##        Logic:
##           1) Show the up to 20 images in a grid pattern as thumbnails.
##           2) User can click on image thumbnail to Select or Deslect an image.
##              By default, initially all the images are Selected.
##           3) Display the current count of Selected images.
##              Change the color of this box (Label) to Green or Red to indicate if within max. limit.
##           2) Provide option to Enlarge image to study more clearly.
##           4) Provide option to perform object detection inference again on an image and see results.
##           5) Allow Confirmation of Deselections by cliking the Confirm butoon.
##              Allow clicking only if current count of Selections <= max limit.
##       Outputs:
##          None
##              Am intermediate file is written and read during logic.
##              Will be automatically deleted as part of execution flow.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -wavlocfile       : location of input file with each line specifying the individual wav files to be processed.
## 2) -opfileallposinfo : location where to write a file with parts-of-speech (POS) info.
## 3) -logfileloc       : location for a log file output
## -----------------------------------------------------------------------------
## Usage example:
##    python3 comb_functional_stt_id_gui_query_img_select_gui_3B.py -wavlocfile "/home/rohit/PyWDUbuntu/thesis/combined_execution/SttTranscribe/stt_wav_files_loc_1.txt" -opfileallposinfo "/home/rohit/PyWDUbuntu/thesis/combined_execution/IdElements/all_words_pos_info_1.txt" -logfileloc "./LOG_comb_functional_stt_id_gui_query_img_select_gui_3B.LOG"
## -----------------------------------------------------------------------------

## import necessary packages

##   imports of common modules
import argparse
import os
import json
import numpy as np
import tkinter as tk
from functools import partial
import logging

##   imports for stt logic
import subprocess

##   imports for identify keywords logic
import spacy
import copy
#from spacy.lang.en.stop_words import STOP_WORDS - not required as using alternative approach

##   imports for neo4j query logic
import sys
from py2neo import Graph

##   imports for neo4j query images selection via gui logic
from PIL import ImageTk, Image
from keras.models import Model as keras_Model, load_model as keras_load_model
import struct
import cv2

## command line arguments
argparser = argparse.ArgumentParser(
    description='pick wav files specified in input file, process via deepspeech and capture transcription output for furhter processing')

argparser.add_argument(
    '-wavlocfile',
    '--wavfilesloccationinputfile',
    help='file containing the location of all the wav files to process')

argparser.add_argument(
    '-opfileallposinfo',
    '--opfileallwordsposinfo',
    help='location for output file where the key elements will be stored')

argparser.add_argument(
    '-logfileloc',
    '--oplogfilelocation',
    help='location for output file for logging')

###############################################################################################################
## -----------------------------  STT LOGIC STARTS              STT LOGIC  STARTS ----------------------------- 
###############################################################################################################
def stt_transcribe_functionality(_wavlocfile, _DEBUG_SWITCH):
    '''
    High level module to execute the STT processing of wav files and get the transcriptions.
    VARIABLES PASSED:
          1) Location of file containing the locations of the wav files to process.
          2) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - containing the transcriptions of each wav file
    '''
    ## check valid input for the -wavlocfile parameter
    ## then check each of the wav files specified
    if not os.path.isfile(_wavlocfile):
        # print(f"\nFATAL ERROR: Input for wavlocfile parameter is not an existing file.\nExiting with RC=100")
        # exit(100)
        return_msg = f"\nFATAL ERROR: Input for wavlocfile parameter is not an existing file.\nRC=100"
        return (100, return_msg, None)
    else:
        try:
            wavfiles_arr = []
            with open(_wavlocfile, 'r') as infile:
                for line in infile:
                    wavfiles_arr.append(line.rstrip("\n"))
            if _DEBUG_SWITCH:
                print(f"\n{wavfiles_arr}\n")
            print(f"\nFollowing wav files are found to process:")
            for idx, each_wav_file in enumerate(wavfiles_arr):
                if not os.path.isfile(each_wav_file):
                    # print(f"\nFATAL ERROR: Check the wav file locations specified in input file.\nThis is not a file:\n{each_wav_file}\nExiting with RC=120")
                    # exit(120)
                    return_msg = f"\nFATAL ERROR: Check the wav file locations specified in input file.\nThis is not a file:\n{each_wav_file}\nRC=120"
                    return (120, return_msg, None)
                else:
                    print(f"\t{idx+1}) {each_wav_file}")
        except Exception as wavlocfile_read_error:
            # print(f"\nFATAL ERROR: Problem reading the input file.\nError message: {wavlocfile_read_error}\nExiting with RC=150")
            # exit(150)
            return_msg = f"\nFATAL ERROR: Problem reading the input file.\nError message: {wavlocfile_read_error}\nRC=150"
            return (150, return_msg, None)
    
    deepspeech_inferences_arr = []
    ## create skeleton command
    ds_inf_cmd_fixed = "deepspeech " + \
                 "--model /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.pbmm " + \
                 "--scorer /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.scorer " + \
                 "--audio " ##/home/rohit/PyWDUbuntu/thesis/audio/wavs/input1.wav - this last part will be added on the fly
    
    if _DEBUG_SWITCH:
        print(f"\n\nds_inf_cmd_fixed=\n{ds_inf_cmd_fixed}\n\n")
    
    ## start model inferencing
    print(f"\n\nCommencing model inference from Deepspeech version 0.7.3.\n")
    for idx, each_wav_file in enumerate(wavfiles_arr): #[:1]:
        ds_inf_cmd = ds_inf_cmd_fixed + each_wav_file
        print(f"\n\n\tCommand number {idx+1}:\n{ds_inf_cmd}")
        inference_run = subprocess.Popen(ds_inf_cmd.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = inference_run.communicate()
        inference_run.wait()
        deepspeech_inferences_arr.append(stdout.rstrip('\n'))
        print(f"\tInference:\n{deepspeech_inferences_arr[-1]}")
    
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - start
    # print(f"\n\ndeepspeech_inferences_arr\n{deepspeech_inferences_arr}\n")
    ## write the inferences to output file
    # try:
    #     with open(opfileloc, 'w') as opfile:
    #         for each_inference in deepspeech_inferences_arr:
    #             opfile.write(each_inference + '.' + '\n')
    # except Exception as opfile_write_error:
    #     print(f"\n\nFATAL ERROR: Problem creating the output file.\nError message: {opfile_write_error}\nExiting with RC=500")
    #     exit(500)
    # print(f"\nOutput file created: {opfileloc}\n")
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - end
    
    return (0, None, deepspeech_inferences_arr)
###############################################################################################################
## -----------------------------  STT LOGIC ENDS                  STT LOGIC  ENDS ----------------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ---------------------------  KEYWORDS LOGIC STARTS        KEYWORDS LOGIC  STARTS --------------------------- 
###############################################################################################################

class c_idkeyelem_wnd_grid_window:

    def __init__(self, _root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
        self.root = _root
        self.sentence_num = _sentence_num
        self.each_sentence_candidate_keywords = _each_sentence_candidate_keywords
        self.index_positions_deselected = _index_positions_deselected
        
        ## array for Buttons - self.buttons_arr_for_words
        ##    later will be populated to become a list of lists
        ##    each inner list will be for a row,
        ##    each element in inner list will be entry for that column in the row
        ##    thus the overall array become a 2 dimensional array of 6 rows x 5 columns
        ##         that can hold up to 30 candidate keywords
        ##    each element will be one label to hold one keyword
        ##
        ##    Button characteristics indicates if its, got no word, or has a Selected word, or a Deselected word
        ##        For a button with a valid word in it - STATE is Normal
        ##        Situation Description        Relief        Borderwidth      Color bg          Color fg         State
        ##        1) Valid word, button        SUNKEN            10            green             white           Normal
        ##             is SELECTED
        ##        2) Valid word, button        RAISED            10            yellow            black           Normal
        ##             is DESELCTED
        ##        3) No word in                FLAT              4             black             white           Disabled
        ##             input data
        self.buttons_arr_for_words = []
        
        self.n_rows_words = 6
        self.n_cols_words = 5
        self.valid_selection_counts = [0, 1, 2, 3]
        self.keywords_selected_count_at_start = len(self.each_sentence_candidate_keywords)
        self.keywords_selected_count_current = self.keywords_selected_count_at_start
        
        self.wnd_grid = tk.Toplevel(master=self.root)
        self.wnd_grid.title(f"Selection Window for Keywords -- Sentence number {self.sentence_num}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text=" ".join( [ "Count of Words currently Selected = ", str(self.keywords_selected_count_at_start) ]),
            relief=tk.FLAT,
            borderwidth=10
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.keywords_selected_count_at_start not in self.valid_selection_counts:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text="Click to Confirm Deselections",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width= ( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        if self.keywords_selected_count_current not in self.valid_selection_counts:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the button array for the grid
        ##    first make skeleton entries for the buttons
        ##    by default assume no word is present to display so all buttons are in disabled state
        ##       with the text saying "No Data"
        for r_idx in range(self.n_rows_words):
            self.wnd_grid.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=50)
            temp_row_data = []
            for c_idx in range(self.n_cols_words):
                temp_row_data.append(
                    tk.Button(
                        master=self.wnd_grid,
                        text="No Data",
                        bg="black", fg="white",
                        relief=tk.FLAT,
                        borderwidth=10,
                        state=tk.DISABLED
                        )
                    )
            self.buttons_arr_for_words.append(temp_row_data)
        
        ## now populate the words and activate where applicable
        idx = 0 ## index to access each keyword from the input list
        for r_idx in range(self.n_rows_words):
            for c_idx in range(self.n_cols_words):
                ## set grid position for all the label elements
                self.buttons_arr_for_words[r_idx][c_idx].grid(
                    row=r_idx, column=c_idx,
                    padx=5, pady=5,
                    sticky="nsew"
                    )
                ## check if word is available from the input array
                if idx < self.keywords_selected_count_at_start:
                    ## Yes, then put the word as the text and by default make the button as Selected
                    self.buttons_arr_for_words[r_idx][c_idx].configure(
                        text=self.each_sentence_candidate_keywords[idx],
                        bg="green", fg="white",
                        relief=tk.SUNKEN,
                        state=tk.NORMAL,
                        command=partial(
                            self.word_button_clicked,
                            r_idx, c_idx
                            )
                        )
                    idx += 1
        
        ## label for Current Count of Selected buttons
        r_idx = self.n_rows_words
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols_words,
            sticky="nsew",
            padx=5, pady=5
            )
        ## label for Confirm Selections
        r_idx = self.n_rows_words + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols_words,
            sticky="nsew",
            padx=5, pady=5
            )
    
    def word_button_clicked(self, _r_idx, _c_idx):

        ## change button characteristics to indicate toggle of Select / Deselect of word
        if self.buttons_arr_for_words[_r_idx][_c_idx]["bg"]=="green":
            ## button is currently Selected, change to Deselected characteristics
            self.buttons_arr_for_words[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                bg="yellow", fg="black"
                )
            self.keywords_selected_count_current -= 1
        else:
            ## button is currently Deselected, change to Selected characteristics
            self.buttons_arr_for_words[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                bg="green", fg="white"
                )
            self.keywords_selected_count_current += 1
        
        ## update the label for count of active selections
        self.lbl_track_selected_count.configure(
            text=" ".join( [ "Count of Words currently Selected = ", str(self.keywords_selected_count_current) ]),
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        ## depending on current count of selections, change the color of the label
        ##    also, disable the button for confirm selections  if required
        if self.keywords_selected_count_current not in self.valid_selection_counts:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
            self.btn_confirm_selections.configure(state=tk.NORMAL, relief=tk.RAISED)
        ## make the Confirm Selections button Disabled if the number of selections is not valid
    
    def do_confirm_selections_processing(self):
        print(f"\n\nCONFIRM SELECTIONS BUTTON PRESSED\n\n")
        ## self.lbl_root_error_message   do something with this later - maybe not required as am disabling
        ##        confirm button if current count is not valid to proceed

        ## For the Deselected keywords, figure out the position and add position number
        ##     to the return list for action later.
        #self.index_positions_deselected
        for r_idx in range(self.n_rows_words):
            for c_idx in range(self.n_cols_words):
                if self.buttons_arr_for_words[r_idx][c_idx]["bg"] == "yellow":
                    self.index_positions_deselected.append( r_idx * self.n_cols_words + c_idx )
        print(f"\nDeselected positions=\n{self.index_positions_deselected}")
        self.root.destroy()

def generic_show_grid_selections_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
    o_idkeyelem_wnd_grid_window = c_idkeyelem_wnd_grid_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected)
    o_idkeyelem_wnd_grid_window.wnd_grid.mainloop()

class c_idkeyelem_root_window:

    def __init__(self, _DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
        self.sentence_num = _sentence_num
        self.each_sentence_candidate_keywords = _each_sentence_candidate_keywords
        ## this is the list passed from beginning. to be populated with positions of deselections during the
        ##      grid window processing
        self.index_positions_deselected = _index_positions_deselected

        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS SENTENCES KEYWORDS!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for the keywords selection for the sentence being processed.",
            f"\n",
            f"\n\t1.  Click on the button below to proceed.",
            f"\n",
            f"\n\t2.  A grid will display showing the candidate keywords as clickable buttons.",
            f"\n\t         By default, ALL the words will be selected at start."
            f"\n\t         Important: If there are no words to display, you cannot click to proceed to"
            f"\n\t                    the grid selection window. Simply close this window and the next"
            f"\n\t                    query selections process will begin."
            f"\n",
            f"\n\t3.  You can toggle the selection of a Keyword by clicking the button for the word.",
            f"\n",
            f"\n\t4.  Once ready, click the button to Confirm Deselections.",
            f"\n\t         NOTE; You can only Select either 1 or 2 or 3 keywords.",
            f"\n\t         Monitor the count of current Selections before confirming Deselections.",
            f"\n",
            f"\n\t5.  Important: If you accidentally close this window, the selection process for next",
            f"\n\t               sentence will start automatically.",
            f"\n"
        ])

        ## root window for the sentence being processed
        self.root = tk.Tk()
        if len(self.each_sentence_candidate_keywords) == 0:
            self.root.title(f"Word selection - Sentence number {_sentence_num} - No Words to Display -- Please close this window")
        else:
            self.root.title(f"Word selection - Sentence number {_sentence_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=30,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are words and make proceed button clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to proceed to selection - Sentence number {_sentence_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10,
            command=partial(
                generic_show_grid_selections_window,
                self.root,
                self.sentence_num,
                self.each_sentence_candidate_keywords,
                self.index_positions_deselected
                )
            )
        ## if no words to display in grid, disable the button to proceed and change the text displayed in button
        if len(self.each_sentence_candidate_keywords) == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No words to make Selections for Sentence number {_sentence_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=100,
            height=7
        )
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

def change_candidate_elements(_DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected):
    '''
    ACTIONS:
        Shows the keywords list for each sentence before any changes are made.
        Accepts user selection for the word positions to be dropped. Validates the values entered.
        Drops the words specified by user selection if allowed to.
        Calls the sanity check function to ensure the remaining words meet the requirements for
              query module in downstream processing.
    ACCEPTS:
        1) Debug switch
        2) Sentence number starting from 1
        3) List of keywords for particular sentence
        4) Array to fill with the deselected positions (starting from 0 index position)
    RETURN:
        1) changes_made_flag : A boolean value indicating if any changes made
                False if no changes required and/ or made by user
                True  if any changes are made
        2) the keywords list (changed or unchanged as per user selection)
    '''
    if _DEBUG_SWITCH:
        ## show keywords before any changes
        print(f"\n\nCandidate key words BEFORE any changes for sentence {_sentence_num} :")
        print(f"{_each_sentence_candidate_keywords}")
        print(f"\n\n")
    
    ## create and show the root window
    o_idkeyelem_root_window = c_idkeyelem_root_window(_DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected)
    o_idkeyelem_root_window.root.mainloop()

def id_elements_functionality(_stt_module_results, _opfileallposinfo, _DEBUG_SWITCH):
    '''
    High level module to execute the identify keywords functionality by processing the 
         transcriptions from the STT functionality.
    VARIABLES PASSED:
          1) Array containing the transcriptions i.e. output of the STT module
          2) Location of file where to write the POS info for all the words that are not stopwords
          3) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - containing the keywords in the required data structure
    '''

    ## We want to create a set of candidate keywords to present to the user as part of gui selection
    ## Start with the transcription sentences from STT block output.
    ##       process to do folllowing for each sentence:
    ##            1) Word tokenize
    ##            2) Drop all stop-words
    ##            3) Do POS tagging, only keep words that are part of the valid type of POS
    ##            4) Take the lemma form of the word, not the original word itself
    ##            5) Only retain word if it appears in the set of object label class names
    ##               It is pointless to present words to the user for selection that are never
    ##               going to return hits in the database query.
    ## 
    ## Now we have the candidate keywords for the setnence for gui selection logic

    ## setup spacy
    nlp = spacy.load('en_core_web_lg')
    POS_INTEREST_INCLUDE_VERBS_SWITCH = False
    
    ## from the input data, if present, remove characters for newline and period
    sentences_orig = [] # sentences to process
    for each_transcription in _stt_module_results:
        sentences_orig.append(each_transcription.rstrip("\n").rstrip("."))
    
    ## create various arrays required
    ## nlp documents array - one per original sentence
    docs = [nlp(each_sentence) for each_sentence in sentences_orig]
    ## array for tokens of the original sentences - create list of word tokens for each sentence doc
    sentences_words_list = [[token.text for token in doc] for doc in docs]
    ## create list of words for each sentence doc - WITHOUT STOP WORDS
    sentences_words_list_no_stop = [[word for word in words_list if not nlp.vocab[word].is_stop ] for words_list in sentences_words_list]
    ## sentences array with stop words removed - only used to display for readability
    sentences_no_stop = [' '.join(words_list) for words_list in sentences_words_list_no_stop]
    
    myStr = f"\n\nThe following sentences will be processed:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, each_input_sentence in enumerate(sentences_orig):
        myStr = f"\tSentence {idx+1} :\n{each_input_sentence}"
        print_and_log(myStr, "info")
        myStr = None
    
    myStr = f"\n\nWords of each input sentence:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, words_list in enumerate(sentences_words_list):
        myStr = f"\tSentence {idx+1} :\n{words_list}"
        print_and_log(myStr, "info")
        myStr = None
    
    myStr = f"\n\nWords of each input sentence after removing all stop words:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, words_list in enumerate(sentences_words_list_no_stop):
        myStr = f"\tSentence {idx+1} :\n{words_list}"
        print_and_log(myStr, "info")
        myStr = None
    
    myStr = f"\n\nJoining the non-stop words as a new sentence (for readability only):\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx, new_sent_no_stop in enumerate(sentences_no_stop):
        myStr = f"\tNew sentence {idx+1} :\n{new_sent_no_stop}"
        print_and_log(myStr, "info")
        myStr = None
    
    ## pos extraction and fill data structure
    pos_info = [[{} for seach_word in words_list] for words_list in sentences_words_list_no_stop]
    for idx1, each_sent_no_stop in enumerate(sentences_no_stop):
        doc = nlp(each_sent_no_stop)
        for idx2, token in enumerate(doc):
            pos_info[idx1][idx2]['text']     = token.text
            pos_info[idx1][idx2]['lemma_']   = token.lemma_
            pos_info[idx1][idx2]['pos_']     = token.pos_
            pos_info[idx1][idx2]['tag_']     = token.tag_
            pos_info[idx1][idx2]['dep_']     = token.dep_
            pos_info[idx1][idx2]['shape_']   = token.shape_
            pos_info[idx1][idx2]['is_alpha'] = token.is_alpha
            pos_info[idx1][idx2]['is_stop']  = token.is_stop
    
    myStr = f"\n\nAll non-stop words pos info:\n"
    print_and_log(myStr, "info")
    myStr = None
    for idx1, each_new_sent in enumerate(pos_info):
        myStr = f"........."
        print_and_log(myStr, "info")
        myStr = None
        for idx2, each_word_pos_info in enumerate(each_new_sent):
            myStr = f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}"
            print_and_log(myStr, "info")
            myStr = None

    ## save the pos info to file for all words that are not stopwords
    try:
        with open(_opfileallposinfo, 'w') as opfileall:
                json.dump(pos_info, opfileall)
        myStr = f"\n\nAll POS info file successfully created here:\n{_opfileallposinfo}\n\n"
        print_and_log(myStr, "info")
        myStr = None
    except Exception as opfileallpos_write_error:
        return_msg = f"\nFATAL ERROR: Problem creating the output file for all words pos info.\nError message: {opfileallpos_write_error}\nRC=200"
        return (200, return_msg, None)
    
    ## extract lemma form of each word that matches a tag of interest.
    ##     note that only Nouns will be considered by default, unless the switch for verbs is True.
    ## tags of interest as per naming convention here: https://spacy.io/api/annotation#pos-tagging
    pos_tags_of_interest_nouns = ['NN', 'NNS']
    pos_tags_of_interest_verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    if POS_INTEREST_INCLUDE_VERBS_SWITCH:
        pos_tags_of_interest = pos_tags_of_interest_nouns + pos_tags_of_interest_verbs
    else:
        pos_tags_of_interest = pos_tags_of_interest_nouns
    
    ## object classes that object detector is trained on
    labels = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', \
            'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', \
            'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', \
            'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', \
            'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', \
            'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', \
            'orange', 'oven', 'parking meter', 'person', 'pizza', 'pottedplant', \
            'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', \
            'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', \
            'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', \
            'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']
    labels_set = set(labels)

    ## array for candidate keywords
    candidate_keywords = []
    for sent_info in pos_info:
        each_sentence_candidate_keywords = []
        for each_word_info in sent_info:
            if each_word_info['tag_'] in pos_tags_of_interest:
                each_sentence_candidate_keywords.append(each_word_info['lemma_'])
        each_sentence_candidate_keywords_set = set(each_sentence_candidate_keywords)
        ## e.g.   {'story', 'fruit', 'banana'} - ( {'story', 'fruit', 'banana'} - {'apple', 'pear', 'banana'} )   =
        ##        {'story', 'fruit', 'banana'} - {'story', 'fruit'}                                               =
        ##        {'banana'}   --> which is what we need finally
        each_sentence_candidate_keywords_labels_matched = list( each_sentence_candidate_keywords_set - (each_sentence_candidate_keywords_set - labels_set) )
        if each_sentence_candidate_keywords_labels_matched: # append to the final list only if its not an empty list
            candidate_keywords.append(each_sentence_candidate_keywords_labels_matched)
    
    myStr = f"\n\nCandidate keywords AFTER matching against class labels:\n{candidate_keywords}\n\n"
    print_and_log(myStr, "info")
    myStr = None

    ## cleanup for gc
    del labels_set, labels
    
    ## execute gui logic to allow deselection of keywords, check the number of selections are valid,
    ##         capture the deselected positions
    index_positions_deselected_all = []
    for sentence_num, each_sentence_candidate_keywords in enumerate(candidate_keywords):
        index_positions_deselected = []
        change_candidate_elements(
            _DEBUG_SWITCH,
            sentence_num + 1,
            each_sentence_candidate_keywords,
            index_positions_deselected
            )
        index_positions_deselected_all.append(index_positions_deselected)
    
    ## extract the Selected keywords into new data structure
    final_keywords = []
    for each_index_positions_deselected, each_sentence_candidate_keywords in \
        zip(index_positions_deselected_all, candidate_keywords):
        temp_arr = [candidate_keyword for idx, candidate_keyword in enumerate(each_sentence_candidate_keywords) if idx not in each_index_positions_deselected]
        final_keywords.append(temp_arr)
    
    for idx, (each_candidate_keywords, each_index_positions_deselected, each_final_keywords) in \
        enumerate(zip(candidate_keywords, index_positions_deselected_all, final_keywords)):
        myStr = "\n".join([
            f"\nSentence {idx+1}:",
            f"BEFORE = {each_candidate_keywords}",
            f"Deselected = {each_index_positions_deselected}",
            f"AFTER = {each_final_keywords}\n"
            ])
        print_and_log(myStr, "debug")
        myStr = None
    
    return (0, None, final_keywords)

###############################################################################################################
## -----------------------------  KEYWORDS LOGIC ENDS        KEYWORDS LOGIC  ENDS ----------------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ----------------------  NEO4J QUERY LOGIC STARTS             NEO4J QUERY LOGIC STARTS ---------------------- 
###############################################################################################################

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
            105         outer list did not contain exacty 1/2/3 elements
            110         some inner element is not a list
            115         some inner list did not contain exactly 0/1/2/3 elements
            120         some element of the inner list is not a string
    '''
    valid_count_words_in_inner_list = [0, 1, 2, 3]
    valid_length_of_outer_list = [1, 2, 3]
    if type(_key_elements_list) != list:
        myStr = f"\n\nERROR 100: Expected a list of lists as the input data. Outer data is not a list. Please check the input and try again.\n"
        print_and_log(myStr, 'error')
        myStr = None
        return 100
    elif len(_key_elements_list) not in valid_length_of_outer_list:
        myStr = f"\n\nERROR 105: Expected a list with exactly one/ two/ three elements. Please check the input and try again.\n"
        print_and_log(myStr, 'error')
        myStr = None
        return 105
    for inner_list in _key_elements_list:
        if type(inner_list) != list:
            myStr = f"\n\nERROR 110: Expected a list of lists as the input data. Inner data is not a list. Please check the input and try again.\n"
            print_and_log(myStr, 'error')
            myStr = None
            return 110
        elif len(inner_list) not in valid_count_words_in_inner_list:
            myStr = f"\n\nERROR 115: Some inner list did not contain expected number of elements. Please check the input and try again.\n"
            print_and_log(myStr, 'error')
            myStr = None
            return 115
        for object_string in inner_list:
            if type(object_string) != str:
                print()
                myStr = f"\n\nERROR 120: Expected a list of lists as the input data, with inner list consisting of strings. Please check the input and try again.\n"
                print_and_log(myStr, 'error')
                myStr = None
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
            1000        unexpected situation
            1500        problem querying the neo4j database
    '''
    ## set result as None by default
    result = None
    
    ## establish connection to Neo4j
    try:
        graph = Graph(uri="bolt://localhost:7687",auth=("neo4j","abc"))
    except Exception as error_msg_neo_connect:
        myStr = "\n".join([
            f"\nUnexpected ERROR connecting to neo4j. Function call return with RC=100. Message:",
            f"{error_msg_neo_connect}\n\n"
            ])
        print_and_log(myStr, 'error')
        myStr = None
        # print(f"\nUnexpected ERROR connecting to neo4j. Function call return with RC=100. Message:\n{error_msg_neo_connect}\n\n")
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
        elif len(_each_inner_list) == 1:
            in_obj1 = _each_inner_list[0]
            result = tx.run(stmt3_one_objects, parameters={'in_obj1': in_obj1, 'in_limit': _in_limit}).data()
        elif len(_each_inner_list) == 0: # possible because user deselected all the keywords
            result = None
        else:
            myStr = "\n".join([
                f"\n\nUnexpected length of the key elements array. Result set to None.",
                f"Function call return with RC=1000.\n\n"
                ])
            print_and_log(myStr, 'error')
            myStr = None
            return (1000, result)
        #tx.commit()
        #while not tx.finished():
        #    pass # tx.finished return True if the commit is complete
    except Exception as error_msg_neo_read:
        myStr = "\n".join([
            f"\n\nUnexpected ERROR querying neo4j.",
            f"Message:\n{error_msg_neo_read}",
            f"Function call return with RC=1500.\n\n"
            ])
        print_and_log(myStr, 'error')
        myStr = None
        return (1500, result)
    ## return tuple of return code and the results. RC = 0 means no errors.
    return (0, result)

def query_neo4j_functionality(_id_elements_module_results ,_query_result_limit, _DEBUG_SWITCH):
    '''
    High level module to execute the querying of database to fetch images based on the keywords identified.
    VARIABLES PASSED:
          1) Array containing the keywords identified as an expected data structure
          2) Value to limit the query result to
          3) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - the database query info
    ''' 
    key_elements_list = copy.deepcopy(_id_elements_module_results)
    
    ### DEBUG -start
    if _DEBUG_SWITCH:
        print(f"\n\ntype key_elements_list = {type(key_elements_list)}\nkey_elements_list =\n{key_elements_list}")
    ### DEBUG - end

    ## perform sanity checks on the keywords data passed
    if check_input_data(key_elements_list):
        # print(f"\nFATAL ERROR: Problem with the data input to neo4j query function. Exiting with return code 300.")
        # exit(300)
        return_msg = f"\nFATAL ERROR: Problem with the data input to neo4j query function. RC=300."
        return (300, return_msg, None)
    
    ## Note: each query to db can return None if there are no words in the query
    query_results_arr = []
    ## query the database with the input data
    #_query_result_limit = 10
    for each_inner_list in key_elements_list:
        query_rc, query_result = query_neo4j_db(each_inner_list, _query_result_limit)
        if query_rc != 0:
            # print(f"\nFATAL ERROR: Problem retrieving data from Neo4j; query_rc = {query_rc}.\nExiting with return code 310.")
            # exit(310)
            return_msg = f"\nFATAL ERROR: Problem retrieving data from Neo4j; query_rc = {query_rc}.\nRC=310."
            return (310, return_msg, None)
        else:
            if _DEBUG_SWITCH:
                print(f"\nQuery result for: {each_inner_list}\n{query_result}\n")
            query_results_arr.append(query_result)
    
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - start
    ## show results
    # print(f"\n\nFinal results=\n{query_results_arr}")
    # print(f"\n\nCOMPLETED NEO4J QUERY LOGIC.\n")
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - end
    
    return (0, None, query_results_arr)

###############################################################################################################
## ----------------------  NEO4J QUERY LOGIC ENDS                 NEO4J QUERY LOGIC ENDS ---------------------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## -----------  QUERY IMAGES SELECTION LOGIC STARTS             QUERY IMAGES SELECTION LOGIC STARTS ----------- 
###############################################################################################################

#global SAVED_KERAS_MODEL_PATH = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'

class c_queryImgSelection_root_window:
    def __init__(self, _DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        ## this is the list passed from beginning. to be populated with positions of deselections during the
        ##      grid window processing
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for candidate image selections for a particular query.",
            f"\n",
            f"\n\t1.  Click on the button to proceed.",
            f"\n",
            f"\n\t2.  A grid displays thumbnails of the candidate images from database query.",
            f"\n\t       Important: If there are no images to display you cannot click to proceed to the"
            f"\n\t                  grid selection window. Simply close this window and the next query"
            f"\n\t                  selections process will begin."
            f"\n",
            f"\n\t3.  By default, all the images are Selected.",
            f"\n\t       You can Deselect any images by clicking the image thumbnail.",
            f"\n\t       You can also Deselect all images.",
            f"\n\t       But you can only Select a maximum of 5 images.",
            f"\n\t       Monitor the count of currently Selected images.",
            f"\n",
            f"\n\t4.  Once ready with your Selections, click the button to Confirm Deselections.", 
            f"\n\t       NOTE: While the number of Selections is invalid (more than 5), you cannot",
            f"\n\t             click the button to confirm Selections.",
            f"\n",
            f"\n\t5.  You can Enlarge an image to inspect it more cloself before deciding to Select or Deselect.",
            f"\n",
            f"\n\t6.  When viewing the Enlarged image, you can also perform object detection on the image to",
            f"\n\t         see an Inference image in a new window."
        ])

        ## root window for the query being processed
        self.root = tk.Tk()
        if _num_candidate_images_before_selection_began == 0:
            self.root.title(f"Image Selection - Query number {_query_num} - No Images to Display -- Please close this window")
        else:
            self.root.title(f"Image Selection - Query number {_query_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are images and make it clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to view images and make Selections - Query {_query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                generic_show_grid_selection_window,
                self.root,
                self.each_query_result,
                self.query_num,
                self.index_positions_to_remove_this_query
                )
            )
        ## if no images to display in grid, disable the button to proceed and change the text displayed in button
        if _num_candidate_images_before_selection_began == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No images to make Selections - Query {_query_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=(len(self.btn_root_click_proceed["text"]) + 10),
            height=7
        )
        
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

def generic_show_grid_selection_window(_root, _each_query_result, _query_num, _index_positions_to_remove_this_query):
    
    ## dictionary with key as the image source, value is the location of that datasets images
    source_and_location_image_datasets = {
        'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
        'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
        'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
        'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
        }
    
    ## build the list for images with their full path
    image_files_list = []
    for each_img_info in _each_query_result:
        image_files_list.append(
            os.path.join(
                source_and_location_image_datasets[each_img_info["Source"]],
                each_img_info["Image"]
                ))
    
    num_candidate_images_in_this_query = len(image_files_list)
    #print(f"Num of images = {num_candidate_images_in_this_query}\narray=\n{image_files_list}")

    opened_imgs_arr = []
    for idx in range(20):
        if idx < num_candidate_images_in_this_query:  ## image list has a valid entry to pick up
            img_orig = Image.open(image_files_list[idx])
            img_resized = img_orig.resize((100, 80),Image.ANTIALIAS)
            ## append tuple of resized image and file with path
            opened_imgs_arr.append( (ImageTk.PhotoImage(img_resized), image_files_list[idx]) )
            del img_orig, img_resized
        else:  ## there is no image
            opened_imgs_arr.append( (None, None) )
    
    ## the selection grid can handle maximum 20 image thumbnails.
    ##     There are 10 columns.
    ##     There are 2 rows of thumbnail images.
    ##     Each row of images has associated Enlarge buttons row below it.
    ##     So the selection window grid will have have 2 x 2 x 10 = 40 buttons.
    ##     In addition to these, there are two more tkinter widgets:
    ##        a Button for Confirm Selections, and
    ##        a Label to show live count of currently Selected images.
    ##     NOTE: We specify the number of rows and columns wrt the images. Logic will later assume
    ##           two additional rows for the Confirm button and Selection count Label.
    n_rows_images = 2
    n_cols_images = 10
    n_rows = 2 * 2 ## one for the images, one for the associated Enlarge button
    n_cols = n_cols_images

    ## make object for object detector - this same object will be used for inferennce on all images
    ##      in the grid. So no need to load new model for each inference.
    o_keras_inference_performer = c_keras_inference_performer()

    o_queryImgSelection_grid_wnd_window = c_queryImgSelection_grid_wnd_window(
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        num_candidate_images_in_this_query,
        n_rows,
        n_cols,
        opened_imgs_arr,  ## REMEMBER this is tuple of (Photo image object resized, image path)
        o_keras_inference_performer
        )
    
    o_queryImgSelection_grid_wnd_window.wnd_grid.mainloop()

class c_queryImgSelection_grid_wnd_window:
    def __init__(
        self,
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        _num_candidate_images_in_this_query,
        _n_rows,
        _n_cols,
        _opened_imgs_arr,
        _o_keras_inference_performer
        ):

        self.root = _root
        self.query_num = _query_num
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        self.num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        self.n_rows = _n_rows   ## NOTE: This only counts the thumbnail and Enlarge rows, not for Confirm button and Label for current count
        self.n_cols = _n_cols
        self.opened_imgs_arr = _opened_imgs_arr
        self.o_keras_inference_performer = _o_keras_inference_performer
        
        ## array for the buttons for thumbnail image button and the associated Enlarge button
        self.grid_buttons_arr = []
        ## The number of Selected candidate images cannot be more than this limit
        self.max_limit_images_selections = 5
        ## initialise the current count of selections to the number of images at start
        self.images_selected_current_count = self.num_candidate_images_in_this_query_at_start

        ## window for the grid selection
        self.wnd_grid = tk.Toplevel(master=_root)
        self.wnd_grid.title(f"Thumbnails and Selection Window -- Query number {self.query_num}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text="",
            relief=tk.FLAT,
            borderwidth=10
            )
        ## call function to update the text with count, and the color 
        self.update_label_count_currently_selected_images()
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text=f"Click to Confirm Deselections -- clickable only if current selection count is NOT > {self.max_limit_images_selections}",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width=( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        ## prevent Confirm Selections if there are too many images Selected at start
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the button array for the grid - thumbnail and Enlarge buttons
        ##    first make skeleton entries for the buttons
        ##    by default assume no Image is present to display,
        ##       so all buttons are in disabled state with the text saying "No Image"
        for r_idx in range(self.n_rows):
            self.wnd_grid.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=50)
            temp_row_data = []
            for c_idx in range(self.n_cols):
                ## alternate row entries for button of image thumbnail and Enlarge
                if r_idx % 2 == 0:
                    ## thumbnail image button
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="No Image",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
                else:
                    ## Enlarge button type
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="Enlarge",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
            self.grid_buttons_arr.append(temp_row_data)

        ## now populate the Images and activate both button where applicable
        img_idx = 0 ## index to access each image from the tuple of (image, path)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.n_cols):
                ## set grid position for all the label elements
                self.grid_buttons_arr[r_idx][c_idx].grid(
                    row=r_idx, column=c_idx,
                    padx=5, pady=5,
                    sticky="nsew"
                )
                ## only for the thumbnail rows, populate the images if it is available
                ##      if yes, change the state of thumbnail and associated Enlarge buttons
                if (r_idx % 2 == 0) and (img_idx < self.num_candidate_images_in_this_query_at_start):
                    ## r_idx is for an image thumbnail row and there is an image to show
                    ## from the input tuple extract the image and the path
                    resized_image, self.grid_buttons_arr[r_idx + 1][c_idx].image_path = self.opened_imgs_arr[img_idx]
                    self.grid_buttons_arr[r_idx][c_idx].image = None
                    self.grid_buttons_arr[r_idx][c_idx].configure(
                            image=resized_image,
                            relief=tk.SUNKEN,
                            borderwidth=10,
                            highlightthickness = 15,
                            highlightbackground = "green", highlightcolor= "green",
                            state=tk.NORMAL,
                            command=partial(
                                self.do_image_select_button_clicked_processing,
                                r_idx, c_idx
                                )
                        )
                    img_idx += 1
                    ## make variable to hold an Enlarged image window object for the associated Enlarge button.
                    ##     set as None for now.
                    ##     if associated Enlarge button is clicked, object will be populated and used.
                    self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window = None
                    ## change the associated Enlarge button
                    self.grid_buttons_arr[r_idx + 1][c_idx].configure(
                            relief=tk.RAISED,
                            borderwidth=10,
                            state=tk.NORMAL,
                            command=partial(
                                generic_show_enlarged_image_window,
                                self.wnd_grid,
                                self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window,
                                self.grid_buttons_arr[r_idx + 1][c_idx].image_path,
                                self.o_keras_inference_performer
                                )
                        )
        
        ## label for count of current selections
        r_idx = self.n_rows
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )

        ## button for selection confirm
        r_idx = self.n_rows + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )
        
        return
    
    def do_confirm_selections_processing(self):
        ## For the images that are Deselected, figure out the position and add the position number
        ##     to the return list.
        self.index_positions_to_remove_this_query
        for r_idx in range(0, self.n_rows, 2):
            for c_idx in range(self.n_cols):
                if self.grid_buttons_arr[r_idx][c_idx]["relief"] == tk.RAISED:
                    ## this image is Deselected - so extract the position
                    self.index_positions_to_remove_this_query.append( ( (r_idx // 2) * self.n_cols ) + c_idx )
        print(f"\nFor Query {self.query_num}, Deselected positions=\n{self.index_positions_to_remove_this_query}")
        self.root.destroy()
        return

    def update_label_count_currently_selected_images(self):
        ## update the count based on latest count of selected images
        ##        also change the color if the count is greater than allowed limit
        self.lbl_track_selected_count.configure(
            text=" ".join([ "Count of Images currently Selected =", str(self.images_selected_current_count) ])
            )
        self.lbl_track_selected_count.configure(
            width=( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        return

    def do_image_select_button_clicked_processing(self, _r_idx, _c_idx):
        ## toggle button characteristics:
        ##                              Relief         Color around image
        ##       Selected   Image       SUNKEN         Green
        ##       Deselected Image       RAISED         Red
        if self.grid_buttons_arr[_r_idx][_c_idx]["relief"] == tk.SUNKEN:
            ## Image is currently Selected, change to Deselected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                highlightbackground = "red", highlightcolor= "red"
            )
            self.images_selected_current_count -= 1
        else:
            ## Image is currently Deselected, change to Selected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                highlightbackground = "green", highlightcolor= "green"
            )
            self.images_selected_current_count += 1
        ## update the label for count
        self.update_label_count_currently_selected_images()
        ## update the confirm button characteristics
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        else:
            self.btn_confirm_selections.configure(state=tk.NORMAL, relief=tk.RAISED)
        return

def generic_show_enlarged_image_window(_wnd_grid, _o_EnlargeImage_window, _image_path, _o_keras_inference_performer):
    ## make the object for Enlarged window and show the window. Currently object is None
    _o_EnlargeImage_window = c_EnlargeImage_window( _wnd_grid, _image_path, _o_keras_inference_performer)
    _o_EnlargeImage_window.wnd_enlarged_img.mainloop()

class c_EnlargeImage_window:
    def __init__(self, _master_wnd, _image_path, _o_keras_inference_performer):
        self.master = _master_wnd
        self.image_path = _image_path
        self.o_keras_inference_performer = _o_keras_inference_performer

        self.img_orig = ImageTk.PhotoImage(Image.open(self.image_path))

        ## window for the Enlarged image
        self.wnd_enlarged_img = tk.Toplevel(master=self.master)
        self.wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.image_path)}")
        ## label for Enlarged image
        self.lbl_enlarged_img = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.FLAT,
            borderwidth=10,
            image=self.img_orig)
        self.lbl_enlarged_img_full_path = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_enlarged_img_full_path.configure(
            width=( len(self.lbl_enlarged_img_full_path["text"]) + 10),
            height=3
            )
        ## button for Inference
        self.btn_enlarged_img_do_inference = tk.Button(
            master=self.wnd_enlarged_img,
            relief=tk.RAISED,
            borderwidth=10,
            text="Click to perform object detection inference on this image",
            bg="yellow", fg="black",
            command=self.do_inference_and_display_output
            )
        self.btn_enlarged_img_do_inference.configure(
            width=( len(self.btn_enlarged_img_do_inference["text"]) + 10 ),
            height=3
            )
        
        self.lbl_enlarged_img.pack(padx=15, pady=15)
        self.lbl_enlarged_img_full_path.pack(padx=15, pady=15)
        self.btn_enlarged_img_do_inference.pack(padx=15, pady=15)
    
    def do_inference_and_display_output(self):
        print(f"\n\nInference invoked for: {self.image_path}")

        ## window for inference
        self.wnd_inference_img = tk.Toplevel(master=self.wnd_enlarged_img)
        self.wnd_inference_img.title(f"Inference for: {os.path.basename(self.image_path)}")
        ## label for the output image after inference - set as None for now
        self.lbl_inference_img = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.FLAT,
            borderwidth=4,
            image=None
            )
        ## label for the image path
        self.lbl_path_img_inferred = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_path_img_inferred.configure(
            width=( len(self.lbl_path_img_inferred["text"]) + 10),
            height=3
            )
        ## label for the inference info of types of objects found
        self.lbl_inference_textual_info = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="blue", fg="white",
            justify='left'
            )
        ## do the actual inference by saved keras model, then superimpose the bounding boxes and write the
        ##    output image to an intermediate file.
        ## But first, make empty list to to be populated by model inference performer.
        ##       entries made will be f-string of name of objects found with percentage
        ##            without the new-line character at end.
        self.objects_found_info_arr = []
        self.o_keras_inference_performer.perform_model_inference(self.image_path, self.objects_found_info_arr)
        ## reload the intermediate file for tkinter to display, then delete the file
        inference_model_output_image_path = r'./intermediate_file_inferenece_image.jpg'
        inference_model_output_img = ImageTk.PhotoImage(Image.open(inference_model_output_image_path))
        os.remove(inference_model_output_image_path)

        ## put the inference outpu image in the label widget
        self.lbl_inference_img.configure(image=inference_model_output_img)
        ## extract the info about objects found and put into the label text
        textual_info_to_display = "\n".join( self.objects_found_info_arr )
        self.lbl_inference_textual_info.configure(text=textual_info_to_display)
        self.lbl_inference_textual_info.configure(
            width=( 20 + max( [len(line_text) for line_text in self.objects_found_info_arr] )  ),
            height=(2 + len(self.objects_found_info_arr) )
            )
        
        ## pack it all
        self.lbl_inference_img.pack(padx=15, pady=15)
        self.lbl_inference_textual_info.pack(padx=15, pady=15)
        self.lbl_path_img_inferred.pack(padx=15, pady=15)

        ## display the Inference window
        self.wnd_inference_img.mainloop()
        return

class c_keras_inference_performer: ## adapted from jbrownlee/keras-yolo3
    
    def __init__(self):
        self.reloaded_yolo_model = None
    
    class BoundBox:
        def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            
            self.objness = objness
            self.classes = classes

            self.label = -1
            self.score = -1
        
        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)
            return self.label
        
        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]  
            return self.score
    
    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        
        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        
        union = w1*h1 + w2*h2 - intersect
        
        return float(intersect) / union
    
    def preprocess_input(self, image, net_h, net_w):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)/new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)/new_h
            new_h = net_h
        
        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image
    
    def decode_netout(self, netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2]  = self._sigmoid(netout[..., :2])
        netout[..., 4:]  = self._sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
            
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                #objectness = netout[..., :4]
                
                if(objectness.all() <= obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                
                box = self.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

                boxes.append(box)
        
        return boxes
    
    def correct_yolo_boxes(self, boxes, image_h, image_w, net_h, net_w):
        if (float(net_w)/image_w) < (float(net_h)/image_h):
            new_w = net_w
            new_h = (image_h*net_w)/image_w
        else:
            new_h = net_w
            new_w = (image_w*net_h)/image_h
        
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
            y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
            
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    
    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0
    
    def draw_boxes(self, image, boxes, labels, obj_thresh, _objects_found_info_arr):
        serial_num_object_box_overall = 1
        for box in boxes:
            label_str = ''
            label = -1
            serial_num_object_box_at_loop_start = serial_num_object_box_overall
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i

                    #print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                    print(f"{serial_num_object_box_overall}) {labels[i]} : {box.classes[i]*100:.2f}%")
                    _objects_found_info_arr.append( f"{serial_num_object_box_overall}) {labels[i]} :\t\t{box.classes[i]*100:.2f} %" )
                    serial_num_object_box_overall += 1
            
            if label >= 0:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
                # cv2.putText(image, 
                #             label_str + ' ' + str(box.get_score()), 
                #             (box.xmin, box.ymin - 13), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 
                #             1e-3 * image.shape[0], 
                #             (0,255,0), 2)
                text_to_superimpose_in_image = f"{serial_num_object_box_at_loop_start}) {label_str}"
                cv2.putText(image, 
                            text_to_superimpose_in_image, 
                            (box.xmin, box.ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
                serial_num_object_box_at_loop_start += 1
        return image
    
    def perform_model_inference(self, _path_infer_this_image, _objects_found_info_arr):
        print(f"\n\nExecuting model inference on image_to_infer : {_path_infer_this_image}\n")
        
        ## load the keras pretained model if not already loaded
        if self.reloaded_yolo_model is None:
            print(f"\n\n    LOADED KERAS MODEL      \n\n")
            saved_model_location = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'
            self.reloaded_yolo_model = keras_load_model(saved_model_location)

        ## set some parameters for network
        net_h, net_w = 416, 416
        obj_thresh, nms_thresh = 0.5, 0.45
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
        ## list of object classes the object detector model is trained on
        labels = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', \
                'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', \
                'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', \
                'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', \
                'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', \
                'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', \
                'orange', 'oven', 'parking meter', 'person', 'pizza', 'pottedplant', \
                'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', \
                'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', \
                'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', \
                'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']
        
        image_to_infer_cv2 = cv2.imread(_path_infer_this_image)
        image_h, image_w, _ = image_to_infer_cv2.shape
        try:
            image_to_infer_preprocessed = self.preprocess_input(image_to_infer_cv2, net_h, net_w)
        except Exception as error_inference_preprocess_image:
            print(f"\nFATAL ERROR: Problem reading the input file.\nError message: {error_inference_preprocess_image}\nExit RC=400")
            exit(400)
        
        ## run the prediction
        yolos = self.reloaded_yolo_model.predict(image_to_infer_preprocessed)
        boxes = []

        for i in range(len(yolos)):
            ## decode the output of the network
            boxes += self.decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        
        ## correct the sizes of the bounding boxes
        self.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        ## suppress non-maximal boxes
        self.do_nms(boxes, nms_thresh)

        ## draw bounding boxes into the image
        self.draw_boxes(image_to_infer_cv2, boxes, labels, obj_thresh, _objects_found_info_arr)

        ## save the image as intermediate file -- see later whether to return and processing is possible
        cv2.imwrite(r'./intermediate_file_inferenece_image.jpg', (image_to_infer_cv2).astype('uint8') )

def select_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
    
    ## make a root window and show it
    o_queryImgSelection_root_window = c_queryImgSelection_root_window(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began)
    o_queryImgSelection_root_window.root.mainloop()

def gui_candidate_image_selection_functionality(_query_neo4j_module_results, _DEBUG_SWITCH):
    '''
    High level module to present images returned by neo4j query via a graphical user interface.
        Allow Deselection of images for two purposes:
            1) images are wrongly picked up (as object of interest is actually not present)
            2) Neo4j query can returnup to 20 images per query. But the maximum limit of images per query
               to pass to the next stage (auto-caption block) is only 5.
        Allow viewing Enlarged image to study clearly. Also, allow object detection inference again.
               Thus provide more information to user to make Deselection decision.
        Show live count of number of images currently Selected (default start is with all images Selected).
        Allow clicking of Deslections Confirmation only if the count of Selected images is within the max.
               permissible limit.
    VARIABLES PASSED:
          1) Data structure containing results of queries to Neo4j database - consists of image name and the
             source name to indicate which dataset the image belongs to.
          2) Debugging switch
    RETURNS:
          1) Return code
                value of 0 => all ok
                non 0      => some problem
          2) Message if return code is non-zero, else will be None
          3) Results array - same data struture as the input but containing only the Selected images.
                Per query, maximum 5 images information will be present.
                NOTE: It could be 0 too, if user decided to Deselect all the images.
    ''' 
    # ## test data for unit test - actually will be passed from earlier stage (id keyword elements)
    # database_query_results = [
    #     [
    #         {'Image': '000000169542.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000169516.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000292186.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000168815.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000168743.jpg', 'Source': 'coco_test_2017'}
    #         ],
    #     [
    #         {'Image': '000000146747.jpg', 'Source': 'coco_test_2017'},
    #         {'Image': '000000509771.jpg', 'Source': 'coco_test_2017'}
    #         ],
    #     [
    #         {'Image': '000000012149.jpg', 'Source': 'coco_test_2017'}
    #         ]
    #     ]
    
    ## execute gui logic to allow deselection of images, check the number of selections are valid,
    ##         capture the deselected positions
    index_positions_to_remove_all_queries = []
    for query_num, each_query_result in enumerate(_query_neo4j_module_results):
        num_candidate_images_before_selection_began = len(each_query_result)
        index_positions_to_remove_this_query = []
        if _DEBUG_SWITCH:
            print(f"\n\nStarting Selection process for Query {query_num + 1}")
            print(f"Number of images before selection began = {num_candidate_images_before_selection_began}\n")
        select_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result, index_positions_to_remove_this_query, num_candidate_images_before_selection_began)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        
        print(f"\nCompleted selection process - Query number {query_num + 1}\n")
        if _DEBUG_SWITCH:
            num_images_to_remove = len(index_positions_to_remove_this_query)
            print(f"\nNumber of images Deselected by user = {num_images_to_remove}.\nNumber of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }")
    
    ## remove the Deselected images
    final_remaining_images_selected_info = []
    for each_query_result, each_index_positions_to_remove in \
        zip(_query_neo4j_module_results, index_positions_to_remove_all_queries):
        temp_array = [each_image_info for idx, each_image_info in enumerate(each_query_result) if idx not in each_index_positions_to_remove]
        final_remaining_images_selected_info.append(temp_array)
    
    ## show summary info
    if _DEBUG_SWITCH:
        print(f"\n\n-------------------------------- SUMMARY INFORMATON --------------------------------")
        for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
            enumerate(zip(_query_neo4j_module_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
            print(f"For Query {query_num + 1}\nNumber of candidate images before selection = {len(each_query_results)}")
            print(f"Number of Deselections done = {len(each_query_index_positions_remove)}")
            print(f"Number of images remaining after Deselections = {len(each_query_final_remaining_images_info)}")
            if _DEBUG_SWITCH:
                print(f"\n\t------ Query images info BEFORE::\n{each_query_results}")
                print(f"\n\t------ Positions removed::\n{each_query_index_positions_remove}")
                print(f"\n\t------ Query images info AFTER::\n{each_query_final_remaining_images_info}\n\n")

    return (0, None, final_remaining_images_selected_info)

###############################################################################################################
## -----------    QUERY IMAGES SELECTION LOGIC ENDS               QUERY IMAGES SELECTION LOGIC ENDS ----------- 
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

def execute_control_logic(args):
    DEBUG_SWITCH = False
    neo4j_query_result_limit = 20

    ## process command line arguments
    wavlocfile        = args.wavfilesloccationinputfile  ## -wavlocfile parameter
    opfileallposinfo  = args.opfileallwordsposinfo       ## -opfileallposinfo parameter
    logfileloc        = args.oplogfilelocation           ## -oplogfilelocation parameter

    ## setup logging file -   levels are DEBUG , INFO , WARNING , ERROR , CRITICAL
    logging.basicConfig(level=logging.DEBUG, filename=logfileloc,                               \
        filemode='w', format='LOG_LEVEL %(levelname)s : %(asctime)s :: %(message)s')

    ## do the STT logic processing
    myStr = "\n".join([
        f"\n\n-------------------------------------------------------------------",
        f"-------------------------------------------------------------------",
        f"              STARTING EXECUTION OF STT LOGIC                      ",
        f"-------------------------------------------------------------------",
        f"-------------------------------------------------------------------\n\n"
        ])
    print_and_log(myStr, "info")
    myStr = None
    ## array for the STT logic results - to hold the filewise transcriptions
    stt_logic_RC, stt_logic_msg, stt_module_results = stt_transcribe_functionality(wavlocfile, DEBUG_SWITCH)

    ## do the identify keywords logic processing
    if stt_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"       STARTING EXECUTION OF IDENTIFY KEYWORDS LOGIC               ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        id_elements_logic_RC, id_elements_logic_msg, id_elements_module_results = id_elements_functionality(stt_module_results, opfileallposinfo, DEBUG_SWITCH)
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: STT logic failed. Return code = {stt_logic_RC}",
            f"\tMessage = {stt_logic_msg}",
            f"\n\nPremature Exit.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(100)

    ## do the query neo4j logic processing
    if id_elements_logic_RC == 0:
        myStr = f"After user selection of candidate keywords logic, RC=0, and results::  id_elements_module_results =\n{id_elements_module_results}"
        print_and_log(myStr, 'info')
        myStr = None

        ## first check all the results are not None - possible if user deselected all the candidate words
        ##       for all the sentences.
        ## in such scenario  there is nothing more to process!!!
        if all(ele== [] for ele in id_elements_module_results):
            myStr = "\n".join([
                f"\n\nUser deselected all words of all sentences during candidate keywords selection stage.",
                f"Nothing left to perform downstream processing\n\n"
                ])
            print_and_log(myStr, 'error')
            return
        
        ## user did not deselect all the words - so continue dowstream processing
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"          STARTING EXECUTION OF QUERY NEO4J LOGIC                  ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        query_neo4j_logic_RC, query_neo4j_logic_msg, query_neo4j_module_results = query_neo4j_functionality(id_elements_module_results, neo4j_query_result_limit, DEBUG_SWITCH)
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Identify keywords logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(200)
  
    ## show the result if no problem - matching candidate images against the keywords
    print(f"\nDatabase query results - Candidate Images at this stage:\n")
    if query_neo4j_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, query_neo4j_module_results)):
            myStr= f"\nQuery {idx+1}) Keywords: {v1}\nQuery result:\n{v2}"
            print_and_log(myStr, "info")
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(300)
    
    ## do the selection culling of candiadte images via gui logic processing
    if query_neo4j_logic_RC == 0:
        myStr = "\n".join([
            f"\n\n-------------------------------------------------------------------",
            f"-------------------------------------------------------------------",
            f"  STARTING EXECUTION OF IMAGE SELECTION VIA GUI                    ",
            f"-------------------------------------------------------------------",
            f"-------------------------------------------------------------------\n\n"
            ])
        print_and_log(myStr, "info")
        myStr = None
        gui_candidate_image_selection_logic_RC, gui_candidate_image_selection_logic_msg, gui_candidate_image_selection_module_results = gui_candidate_image_selection_functionality(query_neo4j_module_results, DEBUG_SWITCH)
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}",
            f"\tMessage = {query_neo4j_logic_msg}",
            f"\nPremature Exit.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        
        print(f"\n\nFATAL ERROR: GUI selection of final images from candidate images logic failed. Return code = {query_neo4j_logic_RC}\n\tMessage = {query_neo4j_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(300)
    
    ## show the result if no problem - matching final selected images against the keywords
    myStr = f"\nImages retained after Deselections (to be passed to Auto-caption block):\n"
    print_and_log(myStr, "info")
    myStr = None
    if gui_candidate_image_selection_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, gui_candidate_image_selection_module_results)):
            myStr = "\n".join([
                f"\n{idx+1}) Keywords: {v1}",
                f"\Selected Images results:\n{v2}",
                f"\nPremature Exit.\n\n"
                ])
            print_and_log(myStr, "info")
            myStr = None
    else:
        myStr = "\n".join([
            f"\n\nFATAL ERROR: GUI based candidate image Deselections failed. Return code = {gui_candidate_image_selection_logic_RC}",
            f"\tMessage = {gui_candidate_image_selection_logic_msg}",
            f"\nPremature Exit.\n\n"
            ])
        print_and_log(myStr, "error")
        myStr = None
        exit(400)
    
    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)

    myStr = f"\n\n\nNormal exit from program.\n"
    print_and_log(myStr, "info")
    myStr = None

    exit(0)