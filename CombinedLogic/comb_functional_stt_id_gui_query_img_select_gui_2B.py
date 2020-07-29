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
##        Write all POS info for these non stopwords to a file (with type of noun or verb).
##        Get user to drop one or more words from the keywords identified.
##        Then using a GUI present the candidate keywords to user for Deselecting words.
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
##              This is because some images may have wrongly found the objects of interest.
##              Each Neo4j query can return up to 20 images. But the maximum number of images per query to
##                   pass to the next stage (Auto caption block) is limited to 5 per query. So allow culling.
##        Logic:
##           1) Show the up to 20 images in a grid pattern as thumbnails.
##           2) Provide option to Enlarge image to study more clearly.
##           3) Provide toggle button to Select/ Deselect an image. By default all images are Selected.
##           4) Provide option to perform object detection inference again on an image and see results.
##           5) Once selections confirmed, ensure total Selections <= max limit (currently 5 per query)
##       Outputs:
##          None
##              Am intermediate file is written and read during logic.
##              Will be automatically deleted as part of execution flow.
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -wavlocfile       : location of input file with each line specifying the individual wav files to be processed.
## 2) -opfileallposinfo : location where to write a file with parts-of-speech (POS) info.
## -----------------------------------------------------------------------------
## Usage example:
##    python3 comb_functional_stt_id_gui_query_img_select_gui_2B.py -wavlocfile "/home/rohit/PyWDUbuntu/thesis/combined_execution/SttTranscribe/stt_wav_files_loc_1.txt" -opfileallposinfo "/home/rohit/PyWDUbuntu/thesis/combined_execution/IdElements/all_words_pos_info_1.txt"
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

    def __init__(self, _root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected, _lbl_root_error_message):
        self.root = _root
        self.sentence_num = _sentence_num
        self.each_sentence_candidate_keywords = _each_sentence_candidate_keywords
        self.index_positions_deselected = _index_positions_deselected
        self.lbl_root_error_message = _lbl_root_error_message
        
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
        self.valid_selection_counts = [1, 2, 3]
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

def generic_show_grid_selections_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected, _lbl_root_error_message):
    o_idkeyelem_wnd_grid_window = c_idkeyelem_wnd_grid_window(_root, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected, _lbl_root_error_message)
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
            f"\nThis is the main control window for a keywords selection for the sentence being processed.",
            f"\n",
            f"\n\t1.  Click on the button below to proceed.",
            f"\n",
            f"\n\t2.  A grid will display showing the candidate keywords as clickable buttons.",
            f"\n\t         By default, ALL the words will be selected at start."
            f"\n",
            f"\n\t3.  You can toggle the selection of a Keyword by clicking the word.",
            f"\n",
            f"\n\t4.  Once ready, click the button to Confirm Deselections.",
            f"\n\t         NOTE; You can only Select either 1 or 2 or 3 keywords.",
            f"\n\t         Monitor the count of current Selections before confirming Deselections.",
            f"\n",
            f"\n\t5.  Important: If you select no words, or more more than 3 wordss and confirm Deselections,",
            f"\n\t               you will have to restart the selection process for the sentence.",
            f"\n",
            f"\n\t                If you accidentally close this window, the selection process for next",
            f"\n\t                sentence will start automatically."
        ])

        self.root = tk.Tk()
        self.root.title(f"Root Window - Sentence number {_sentence_num}")
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=30,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message = tk.Label(
            master=self.root,
            text="No Errors detected so far.",
            bg="green", fg="white",
            justify=tk.LEFT,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message.configure(
            width=( len(self.lbl_root_error_message["text"]) + 60 ),
            height=3
            )
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to proceed to selection for Sentence number {_sentence_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10,
            command=partial(
                generic_show_grid_selections_window,
                self.root,
                self.sentence_num,
                self.each_sentence_candidate_keywords,
                self.index_positions_deselected,
                self.lbl_root_error_message
                )
            )
        self.btn_root_click_proceed.configure(
            width=100,
            height=7
        )
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.lbl_root_error_message.pack(padx=10, pady=10)
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
    ## array for tokens of original sentence after removing stop words
    sentences_words_list = [[token.text for token in doc] for doc in docs]
    ## create list of words for each sentence doc - WITHOUT STOP WORDS
    sentences_words_list_no_stop = [[word for word in words_list if not nlp.vocab[word].is_stop ] for words_list in sentences_words_list]
    ## sentences array with stop words removed - only used to display for readability
    sentences_no_stop = [' '.join(words_list) for words_list in sentences_words_list_no_stop]
    
    print(f"\n\nThe following sentences will be processed:\n")
    for idx, each_input_sentence in enumerate(sentences_orig):
        print(f"\tSentence {idx+1} :\n{each_input_sentence}")

    if _DEBUG_SWITCH:
        print(f"\n\nEach input sentence broken into words:\n")
        for idx, words_list in enumerate(sentences_words_list):
            print(f"\tSentence {idx+1} :\n{words_list}")
    
    print(f"\n\nWords of each input sentence after removing any stop words:\n")
    for idx, words_list in enumerate(sentences_words_list_no_stop):
        print(f"\tSentence {idx+1} :\n{words_list}")
    
    if _DEBUG_SWITCH:
        print(f"\n\nJoining the non-stop words as a new sentence (for readability only):\n")
        for idx, new_sent_no_stop in enumerate(sentences_no_stop):
            print(f"\tNew sentence {idx+1} :\n{new_sent_no_stop}")
    
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
    
    if _DEBUG_SWITCH:
        print(f"\n\nAll non-stop words pos info:\n")
        for idx1, each_new_sent in enumerate(pos_info):
            print(f".........")
            for idx2, each_word_pos_info in enumerate(each_new_sent):
                print(f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}")

    ## save the pos info to file for all words that are not stopwords
    try:
        with open(_opfileallposinfo, 'w') as opfileall:
                json.dump(pos_info, opfileall)
        print(f"\n\nAll POS info file successfully created here:\n{_opfileallposinfo}\n\n")
    except Exception as opfileallpos_write_error:
        # print(f"\nFATAL ERROR: Problem creating the output file for all words pos info.\nError message: {opfileallpos_write_error}\nExiting with RC=200")
        # exit(200)
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
    
    candidate_keywords = []
    for sent_info in pos_info:
        each_sentence_candidate_keywords = []
        for each_word_info in sent_info:
            if each_word_info['tag_'] in pos_tags_of_interest:
                each_sentence_candidate_keywords.append(each_word_info['lemma_'])
        if each_sentence_candidate_keywords: # append to the final list only if its not an empty list
            candidate_keywords.append(each_sentence_candidate_keywords)
    
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
    for idx1, (each_index_positions_deselected, each_sentence_candidate_keywords) in \
        enumerate(zip(index_positions_deselected_all, candidate_keywords)):
        temp_arr = [candidate_keyword for idx, candidate_keyword in enumerate(each_sentence_candidate_keywords) if idx not in each_index_positions_deselected]
        final_keywords.append(temp_arr)
    
    if _DEBUG_SWITCH:
        for idx, (each_candidate_keywords, each_index_positions_deselected, each_final_keywords) in \
            enumerate(zip(candidate_keywords, index_positions_deselected_all, final_keywords)):
            print(f"\nSentence {idx+1}:\nBEFORE = {each_candidate_keywords}")
            print(f"Deselected = {each_index_positions_deselected}\nAFTER = {each_final_keywords}\n")
    
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
        print(f"\nUnexpected ERROR connecting to neo4j. Function call return with RC=100. Message:\n{error_msg_neo_connect}\n\n")
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

class c_my_root_window:
    def __init__(self, _query_num, _each_query_result, _func_show_grid_selection_window):
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for a particular query.",
            f"\n",
            f"\n\t1.  Click on the button to proceed.",
            f"\n",
            f"\n\t2.  A grid displays for the candidate images from database query.",
            f"\n",
            f"\n\t3.  You can Deselect any images you want by toggling the Select button.",
            f"\n",
            f"\n\t4.  You can also Deselect all images.",
            f"\n",
            f"\n\t5.  But you can only Select maximum 5 images.",
            f"\n",
            f"\n\t6.  Once ready, you can click the button to Confirm Deselections.",
            f"\n",
            f"\n\t7.  Monitor the count of currently Selected images before confirming Deselections.",
            f"\n",
            f"\n\t8.  You can Enlarge an image to inspect it more cloself before deciding to Select or Deselect.",
            f"\n",
            f"\n\t9.  When viewing the Enlarged image, you can also perform image inference to show an"
            f"\n\t         Inference image in a new window with the objects detected.",
            f"\n",
            f"\n\t10.  Important: If you select more than 5 images and confirm Deselections,",
            f"\n\t                you will have to start the whole process of selection again.",
            f"\n",
            f"\n\t                If you accidentally close this window, the next query selection",
            f"\n\t                will start off."
        ])
        ## to hold deselected positions info. Eventually the inner list will
        ##    be replaced by an interget list of positions
        self.done_button_deselected_positions_results = [[]]
        self.func_show_grid_selection_window = _func_show_grid_selection_window

        self.root = tk.Tk()
        self.root.title(f"Root Window - Query number {_query_num}")
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message = tk.Label(
            master=self.root,
            text="No Errors detected so far.",
            bg="green", fg="white",
            justify=tk.LEFT,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message.configure(
            width=( len(self.lbl_root_error_message["text"]) + 60 ),
            height=3
            )
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to proceed to selection for Query {_query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10,
            command=partial(
                self.func_show_grid_selection_window,
                self.root,
                self.each_query_result,
                self.query_num,
                self.done_button_deselected_positions_results,
                self.lbl_root_error_message
                )
            )
        self.btn_root_click_proceed.configure(
            width=100,
            height=7
        )
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.lbl_root_error_message.pack(padx=10, pady=10)
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

class c_my_wnd_main_window:
    def __init__(self, _root, _query_num, _root_done_button_deselected_positions_results, _num_candidate_images_in_this_query, _lbl_root_error_message, _o_keras_inference_performer):
        self.wnd = tk.Toplevel(master=_root)
        self.root = _root
        self.query_num = _query_num
        self.frame_arr = None
        self.frame_arr_n_rows = None
        self.frame_arr_n_cols = None
        self.o_keras_inference_performer = _o_keras_inference_performer
        ## after user makes deselections, the number of candidate images cannot be more than this limit
        self.max_limit_images_remaining_after_deselections = 5
        self._num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        self.lbl_root_error_message = _lbl_root_error_message
        self._root_done_button_deselected_positions_results = _root_done_button_deselected_positions_results
        self.frm_done = tk.Frame(
            master=self.wnd,
            relief=tk.FLAT,
            borderwidth=10
            )
        self.btn_done = tk.Button(master=self.frm_done, text="Click to Confirm Deselections",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_button_done_processing
            )
        self.btn_done.configure(
            width= ( len(self.btn_done["text"]) + 20 ),
            height=5
            )
        self.frm_track_selected_count = tk.Frame(
                    master=self.wnd,
                    relief=tk.FLAT,
                    borderwidth=10
                    )
        self.lbl_track_selected_count = tk.Label(
            master=self.frm_track_selected_count,
            text=" ".join( [ "Count of Images currently Selected =", str(self._num_candidate_images_in_this_query_at_start) ]),
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self._num_candidate_images_in_this_query_at_start > 5:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        self.wnd.title(f"Thumbnails and Selection Window -- Query number {_query_num}")
    
    def do_button_done_processing(self):
        ## For the images that are Deselected, figure out the position and add position number
        ##     to the return list for action later.
        index_positions_of_deselected_result = []
        for r_idx in range(2, self.frame_arr_n_rows, 3):
            for c_idx in range(self.frame_arr_n_cols):
                if self.frame_arr[r_idx][c_idx].btn_select["text"] == "Deselected":
                    index_positions_of_deselected_result.append( ( (r_idx // 3) * self.frame_arr_n_cols ) + c_idx )
        #print(f"\nDeselected positions=\n{index_positions_of_deselected_result}")

        ## check the number of deselections meets requirements for logic:
        ##       maximum selections remaining is 5 or less.
        ##       Note: User can deselect ALL images i.e. No images are ok as per user.
        num_images_remaining_after_deselections = self._num_candidate_images_in_this_query_at_start - len(index_positions_of_deselected_result)
        if num_images_remaining_after_deselections > self.max_limit_images_remaining_after_deselections:
            ## problem: candidate images remaning greater than maximum limit
            print(f"\nPROBLEM: For {self.query_num}: After Deselections, number of images remaining = {num_images_remaining_after_deselections}, which is greater than the maximum limit allowed = {self.max_limit_images_remaining_after_deselections}\n")
            print(f"\nYou need to restart the selection process....\n")
            ## change the color and message in the Error message Label in the root
            msg_text_for_root_error_label = "".join([
                f"ERROR: After Deselections, number of images remaining = {num_images_remaining_after_deselections},",
                f"\n     which is greater than the maximum limit = {self.max_limit_images_remaining_after_deselections}",
                f"\nYou need to restart the selection process...."
                ])
            self.lbl_root_error_message["text"] = msg_text_for_root_error_label
            self.lbl_root_error_message.configure(bg="red", fg="white", height=8, width=80)
            self.lbl_root_error_message.pack(padx=20, pady=20)
            self.wnd.destroy()
        else:
            ## all good so proceed to return information and destroy the root window
            self._root_done_button_deselected_positions_results[0] = index_positions_of_deselected_result
            self.root.destroy()
    
    def build_frames(self, _n_rows, _n_cols, _in_img_arr_resized):
        ## the grid will always accomodate 20 images. The incoming array has tuple of the resized image and the
        ##     absolute path to image. But if the incoming array has less images, these entries will be None type.

        ## create array holding skeleton form of the objects of required frame types
        self.frame_arr = []
        self.frame_arr_n_rows = _n_rows
        self.frame_arr_n_cols = _n_cols
        for r_idx in range(_n_rows):
            self.wnd.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd.rowconfigure(r_idx, weight=1, minsize=50)
            temp_array = []
            for c_idx in range(_n_cols):
                frame_name_str = ''.join( ['frm_', str(r_idx+1), '_', str(c_idx+1)] )
                ## picture frame
                if (r_idx % 3) == 0:
                    temp_array.append(my_frm_image(self.wnd, frame_name_str))
                ## enlarge frame
                elif (r_idx % 3) == 1:
                    temp_array.append(my_frm_enlarge(self.wnd, frame_name_str, self.o_keras_inference_performer))
                ## select frame
                else:
                    temp_array.append(my_frm_select(self.wnd, frame_name_str, self.lbl_track_selected_count))
            self.frame_arr.append(temp_array)
        
        ## set the frame arrays objects as required
        img_idx = 0
        for r_idx in range(_n_rows):
            for c_idx in range(_n_cols):
                ## picture frame
                if (r_idx) % 3 == 0:
                    self.frame_arr[r_idx][c_idx].frm_img.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    ## populate the image or set it saying "No Image"
                    resized_image, file_with_path = _in_img_arr_resized[img_idx]
                    if resized_image is not None:
                        self.frame_arr[r_idx][c_idx].add_image(resized_image)
                        img_idx += 1
                        self.frame_arr[r_idx][c_idx].image_with_path = file_with_path
                    else:
                        self.frame_arr[r_idx][c_idx].image_with_path = None
                        self.frame_arr[r_idx][c_idx].lbl_img = tk.Label(
                            master=self.frame_arr[r_idx][c_idx].frm_img,
                            text=f"No Image",
                            bg="black", fg="white",
                            width=12, height=4
                            )
                        self.frame_arr[r_idx][c_idx].lbl_img.pack(padx=5, pady=5)
                ## enlarge frame
                elif (r_idx % 3) == 1:
                    self.frame_arr[r_idx][c_idx].frm_enlarge.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    self.frame_arr[r_idx][c_idx].enlarge_this_image = self.frame_arr[r_idx - 1][c_idx].image_with_path ## bcoz immediately previous row should reference the image frame
                    ## disable the Enlarge button if there is no associated image
                    if self.frame_arr[r_idx][c_idx].enlarge_this_image is None:
                        self.frame_arr[r_idx][c_idx].btn_enlarge.configure(state=tk.DISABLED)
                    self.frame_arr[r_idx][c_idx].btn_enlarge.pack(padx=5, pady=5)
                ## select frame
                else:
                    self.frame_arr[r_idx][c_idx].frm_select.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    ## disable the Select button if there is no associated image
                    if self.frame_arr[r_idx-2][c_idx].image_with_path is None:
                        self.frame_arr[r_idx][c_idx].btn_select.configure(state=tk.DISABLED)
                    self.frame_arr[r_idx][c_idx].btn_select.pack(padx=5, pady=5)
        
        ## the final button for submitting selected images
        r_idx = _n_rows
        c_idx = 0
        self.frm_done.grid(row=r_idx, column=c_idx, rowspan=1, columnspan=_n_cols, sticky="nsew", padx=5, pady=5)
        self.btn_done.pack(padx=10, pady=10, fill='both', expand=True)

        ## label to show how many images are currently selected
        r_idx = _n_rows + 1
        c_idx = 0
        self.frm_track_selected_count.grid(row=r_idx, column=c_idx, rowspan=1, columnspan=_n_cols, sticky="nsew", padx=5, pady=5)
        self.lbl_track_selected_count.pack(padx=10, pady=10, fill='both', expand=True)

class my_frm_image:
    def __init__(self, _wnd, _frame_name):
        
        self.frm_img = tk.Frame(
            master=_wnd,
            relief=tk.SUNKEN,
            borderwidth=2
            )
        self.frame_name = _frame_name
        self.image_with_path = None
        self.lbl_img = tk.Label(master=self.frm_img, image=None)
    
    def add_image(self, _in_img):
        self.lbl_img.configure(image=_in_img)
        self.lbl_img.pack(padx=1, pady=1)

class my_frm_enlarge:
    def __init__(self, _wnd, _frame_name, _o_keras_inference_performer):
        self.o_keras_inference_performer = _o_keras_inference_performer
        self.frm_enlarge = tk.Frame(
            master=_wnd,
            relief=tk.RAISED,
            borderwidth=4
            )
        self.master = _wnd
        self.frame_name = _frame_name
        self.enlarge_this_image = None
        self.btn_enlarge = tk.Button(
            master=self.frm_enlarge,
            text=f"Enlarge",
            bg="black", fg="white", 
            command=self.do_enlarge_btn_press_functionality
            )
        self.btn_enlarge.pack(padx=5, pady=5)
        ## enlarged image related - set as None at start
        self.wnd_enlarged_img = None
        self.lbl_enlarged_img = None
        self.lbl_enlarged_img_full_path = None
        self.btn_enlarged_img_do_inference = None
        ## inference image related - set as None at start
        self.wnd_inference_img = None
        self.lbl_inference_img = None
        self.lbl_path_img_inferred = None
        self.lbl_inference_textual_info = None
        ## empty list to to be populated later by model inference performer.
        ##       entry is f-string of object found with percentage; no new-line character at end.
        self.text_info_arr_for_label = []
    
    def do_inference_and_display_output(self):
        print(f"\n\nInference invoked for: {self.enlarge_this_image}")
        self.wnd_inference_img = tk.Toplevel(master=self.wnd_enlarged_img)
        self.wnd_inference_img.title(f"Inference for: {os.path.basename(self.enlarge_this_image)}")
        self.lbl_inference_img = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=2,
            image=None
            )
        
        ## do the actual inference by saved keras model, then superimpose the bounding boxes and write the
        ##    output image to an intermediate file
        self.o_keras_inference_performer.perform_model_inference(self.enlarge_this_image, self.text_info_arr_for_label)

        ## reload the intermediate file for tkinter to display
        inference_model_output_image_path = r'./intermediate_file_inferenece_image.jpg'
        inference_model_output_image = ImageTk.PhotoImage(Image.open(inference_model_output_image_path))
        self.lbl_inference_img.configure(
            image=inference_model_output_image
            )
        self.lbl_path_img_inferred = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image full path = {self.enlarge_this_image}"
            )
        self.lbl_path_img_inferred.configure(
            width=( len(self.lbl_path_img_inferred["text"]) + 10),
            height=3
            )
        textual_info_to_display = "\n".join( self.text_info_arr_for_label )
        self.lbl_inference_textual_info = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="blue", fg="white",
            justify='left',
            text=textual_info_to_display
            )
        
        self.lbl_inference_textual_info.configure(
            width=( 20 + max( [len(line_text) for line_text in self.text_info_arr_for_label] )  ),
            height=(2 + len(self.text_info_arr_for_label) )
            )
        self.lbl_inference_img.pack(padx=15, pady=15)
        self.lbl_inference_textual_info.pack(padx=15, pady=15)
        self.lbl_path_img_inferred.pack(padx=15, pady=15)
        
        self.wnd_inference_img.mainloop()

    def do_enlarge_btn_press_functionality(self):
        #print(f"Would have enlarged image: {self.enlarge_this_image}")
        img_orig = ImageTk.PhotoImage(Image.open(self.enlarge_this_image))
        self.wnd_enlarged_img = tk.Toplevel(master=self.master)
        self.wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.enlarge_this_image)}")
        self.lbl_enlarged_img = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=10,
            image=img_orig)
        self.lbl_enlarged_img_full_path = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image full path = {self.enlarge_this_image}"
            )
        self.lbl_enlarged_img_full_path.configure(
            width=( len(self.lbl_enlarged_img_full_path["text"]) + 10),
            height=3
            )
        self.btn_enlarged_img_do_inference = tk.Button(
            master=self.wnd_enlarged_img,
            relief=tk.RAISED,
            borderwidth=10,
            text="Click to run Inference and show images with detected objects",
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
        self.wnd_enlarged_img.mainloop()

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
    
    def draw_boxes(self, image, boxes, labels, obj_thresh, _inference_text_info_arr):
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
                    _inference_text_info_arr.append( f"{serial_num_object_box_overall}) {labels[i]} :\t\t{box.classes[i]*100:.2f} %" )
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
        
    def perform_model_inference(self, _path_infer_this_image, _inference_text_info_arr):
        print(f"\n\nExecuting model inference on image_to_infer : {_path_infer_this_image}\n")
        
        ## load the keras pretained model if not already loaded
        if self.reloaded_yolo_model is None:
            #print(f"\n\n    LOADED KERAS MODEL      \n\n")
            saved_model_location = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'
            self.reloaded_yolo_model = keras_load_model(saved_model_location)

        ## set some parameters for network
        net_h, net_w = 416, 416
        obj_thresh, nms_thresh = 0.5, 0.45
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
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
        self.draw_boxes(image_to_infer_cv2, boxes, labels, obj_thresh, _inference_text_info_arr)

        ## save the image as intermediate file -- see later whether to return and processing is possible
        cv2.imwrite(r'./intermediate_file_inferenece_image.jpg', (image_to_infer_cv2).astype('uint8') )

class my_frm_select:
    def __init__(self, _wnd, _frame_name, _lbl_track_selected_count):
        self._lbl_track_selected_count = _lbl_track_selected_count
        self.frm_select = tk.Frame(
            master=_wnd,
            relief=tk.RAISED,
            borderwidth=4)
        self.frame_name = _frame_name
        self.btn_select = tk.Button(
            master=self.frm_select,
            text=f"Selected",
            bg="black", fg="white",
            command=self.do_select_btn_press_functionality
        )
        self.btn_select.pack(padx=5, pady=5)
    
    def do_select_btn_press_functionality(self):
        #print(f"Pressed Select button: {self.frame_name}")
        ## get the current count
        current_count_selected = int( self._lbl_track_selected_count["text"].split()[-1] )
        fixed_message = " ".join( self._lbl_track_selected_count["text"].split()[:-1] )
        if self.btn_select["text"] == 'Selected':
            ## Sink button and change text to Deselect
            self.btn_select.configure(text=f"Deselected", relief=tk.SUNKEN, bg="yellow", fg="black")
            current_count_selected -= 1
        else:
            ## Raise button and change text to Select
           self.btn_select.configure(text=f"Selected", relief=tk.RAISED, bg="black", fg="white")
           current_count_selected += 1
        self._lbl_track_selected_count["text"] = " ".join( [fixed_message, str(current_count_selected)] )
        if current_count_selected > 5:
            self._lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self._lbl_track_selected_count.configure(bg="green", fg="white")
        return

def show_grid_selection_window(_root, _each_query_result, _query_num, _root_done_button_deselected_positions_results, _lbl_root_error_message):
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
    ##     Each row of image has associated Enlarge row and Select row i.e. three rows 
    ##          logically show information for each row of actual images.
    ##     So the selection window grid will have have 2 x 3 x 10 = 60 frames.
    ##  NOTE: WHILE BUILDING THE WINDOW, THE LOGIC AUTOMATICALLY CREATES TWO ADDITIONAL ROWS - CONFIRM BUTTON AND COUNT LABEL.

    n_rows = 2 * 3
    n_cols = 10
    #print(f"n_rows = {n_rows}\t\tn_cols = {n_cols}\n")

    o_keras_inference_performer = c_keras_inference_performer()
    o_main_window = c_my_wnd_main_window(_root, _query_num, _root_done_button_deselected_positions_results, num_candidate_images_in_this_query, _lbl_root_error_message, o_keras_inference_performer)
    o_main_window.build_frames(n_rows, n_cols, opened_imgs_arr)
    o_main_window.wnd.mainloop()

def select_candidate_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result):
    
    ## make a root window and show it
    o_root_window = c_my_root_window(_query_num, _each_query_result, show_grid_selection_window)
    o_root_window.root.mainloop()
    o_root_window.done_button_deselected_positions_results
    if _DEBUG_SWITCH:
        print(f"\nQuery {_query_num} :::  o_root_window.done_button_deselected_positions_results = {o_root_window.done_button_deselected_positions_results}")
    return o_root_window.done_button_deselected_positions_results[0]

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
        Once final confirmation is done, verify max. 5 images Selected, else get user to start over for
               to start over for that query.
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
    
    final_remaining_images_selected_info = []
    index_positions_to_remove_all_queries = []
    
    for query_num, each_query_result in enumerate(_query_neo4j_module_results): #(database_query_results):
        temp_array = []
        if _DEBUG_SWITCH:
            print(f"\n\nStarting Selection process for Query {query_num + 1}")
            num_candidate_images_before_selection_began = len(each_query_result)
            print(f"Number of images before selection began = {num_candidate_images_before_selection_began}\n")
        index_positions_to_remove_this_query = select_candidate_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        num_images_to_remove = len(index_positions_to_remove_this_query)
        if _DEBUG_SWITCH:
            print(f"\nNumber of images Deselected by user = {num_images_to_remove}.\nNumber of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }")
        ## remove the Deselected images
        for idx_each_image, each_image_info in enumerate(each_query_result):
            if idx_each_image not in index_positions_to_remove_this_query:
                temp_array.append(each_image_info)
        final_remaining_images_selected_info.append(temp_array)
        if _DEBUG_SWITCH:
            print(f"\nCompleted selection process - Query number {query_num + 1}\n")
    
    ## show summary info
    print(f"\n\n-------------------------------- SUMMARY INFORMATON STARTS -------------------------------\n")
    # for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
    #     enumerate(zip(database_query_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
    for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
        enumerate(zip(_query_neo4j_module_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
        print(f"For Query {query_num + 1}\nNumber of candidate images before selection = {len(each_query_results)}")
        print(f"Number of Deselections done = {len(each_query_index_positions_remove)}")
        print(f"Number of images remaining after Deselections = {len(each_query_final_remaining_images_info)}")
        if _DEBUG_SWITCH:
            print(f"\n\t------ Query images info BEFORE::\n{each_query_results}")
            print(f"\n\t------ Positions removed::\n{each_query_index_positions_remove}")
            print(f"\n\t------ Query images info AFTER::\n{each_query_final_remaining_images_info}\n\n")
    print(f"\n\n-------------------------------- SUMMARY INFORMATON ENDS ---------------------------------\n")

    #print(f"\n\nNormal exit.\n\n")
    return (0, None, each_query_final_remaining_images_info)

###############################################################################################################
## -----------    QUERY IMAGES SELECTION LOGIC ENDS               QUERY IMAGES SELECTION LOGIC ENDS ----------- 
###############################################################################################################

## ----------------------------------------------------------------------------------------------------------------------

###############################################################################################################
## ----------------------  CONTROL LOGIC STARTS                    CONTROL LOGIC STARTS ---------------------- 
###############################################################################################################

def execute_control_logic(args):
    DEBUG_SWITCH = False
    neo4j_query_result_limit = 20

    ## process command line arguments
    wavlocfile        = args.wavfilesloccationinputfile  ## -wavlocfile parameter
    opfileallposinfo  = args.opfileallwordsposinfo       ## -opfileallposinfo parameter

    ## do the STT logic processing
    print(f"\n\n-------------------------------------------------------------------")
    print(f"-------------------------------------------------------------------")
    print(f"              STARTING EXECUTION OF STT LOGIC                      ")
    print(f"-------------------------------------------------------------------")
    print(f"-------------------------------------------------------------------\n\n")
    ## array for the STT logic results - to hold the filewise transcriptions
    stt_logic_RC, stt_logic_msg, stt_module_results = stt_transcribe_functionality(wavlocfile, DEBUG_SWITCH)

    ## do the identify keywords logic processing
    if stt_logic_RC == 0:
        print(f"\n\n-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------")
        print(f"       STARTING EXECUTION OF IDENTIFY KEYWORDS LOGIC               ")
        print(f"-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------\n\n")
        id_elements_logic_RC, id_elements_logic_msg, id_elements_module_results = id_elements_functionality(stt_module_results, opfileallposinfo, DEBUG_SWITCH)
    else:
        print(f"\n\nFATAL ERROR: STT logic failed. Return code = {stt_logic_RC}\n\tMessage = {stt_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(100)

    ## do the query neo4j logic processing
    if id_elements_logic_RC == 0:
        print(f"\n\n-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------")
        print(f"          STARTING EXECUTION OF QUERY NEO4J LOGIC                  ")
        print(f"-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------\n\n")
        query_neo4j_logic_RC, query_neo4j_logic_msg, query_neo4j_module_results = query_neo4j_functionality(id_elements_module_results, neo4j_query_result_limit, DEBUG_SWITCH)
    else:
        print(f"\n\nFATAL ERROR: Identify keywords logic failed. Return code = {query_neo4j_logic_RC}\n\tMessage = {query_neo4j_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(200)
  
    ## show the result if no problem - matching candidate images against the keywords
    print(f"\nDatabase query results - Candidate Images at this stage:\n")
    if query_neo4j_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, query_neo4j_module_results)):
            print(f"\nQuery {idx+1}) Keywords: {v1}\nQuery result:\n{v2}")
    else:
        print(f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}\n\tMessage = {query_neo4j_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(300)
    
    ## do the selection culling of candiadte images via gui logic processing
    if query_neo4j_logic_RC == 0:
        print(f"\n\n-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------")
        print(f"  STARTING EXECUTION OF IMAGE SELECTION VIA GUI                    ")
        print(f"-------------------------------------------------------------------")
        print(f"-------------------------------------------------------------------\n\n")
        gui_candidate_image_selection_logic_RC, gui_candidate_image_selection_logic_msg, gui_candidate_image_selection_module_results = gui_candidate_image_selection_functionality(query_neo4j_module_results, DEBUG_SWITCH)
    else:
        print(f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}\n\tMessage = {query_neo4j_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(300)
    
    ## show the result if no problem - matching final selected images against the keywords
    print(f"\nImages retained after Deselections (to be passed to Auto-caption block):\n")
    if gui_candidate_image_selection_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, gui_candidate_image_selection_module_results)):
            print(f"\n{idx+1}) Keywords: {v1}\nSelected Images results:\n{v2}")
    else:
        print(f"\n\nFATAL ERROR: GUI based candidate image Deselections failed. Return code = {gui_candidate_image_selection_logic_RC}\n\tMessage = {gui_candidate_image_selection_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(400)
    
    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)
    print(f"\n\n\nNormal exit from program.\n")
    exit(0)