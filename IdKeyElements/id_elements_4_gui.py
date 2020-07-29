## -----------------------------------------------------------------------------
## Goal:  Process the input file which has the speech to text transcriptions (one per line).
###       Using Spacy, remove stop-words, perform part of speech tagging.
###       Get user to drop one or more words from the keywords identified.
###       Perform sanity check on the remaning keywords (after dropping any words specified by user)
###            to ensure it satisfies requirements for the neo4j query module.
###       Output two files:
###          One file has: for ALL the words, their comprehensive nlp info from Spacy.
###          Second file has: the data strucutre required for neo4j query logic.
###               Depending on the value of the boolean variable POS_INTEREST_INCLUDE_VERBS_SWITCH
##                either only nouns, or both nouns and verbs, will be considered during processing.
## -----------------------------------------------------------------------------
##  Spacy gives following info for each word as part of POS processing
##    Text: The original word text.
##    Lemma: The base form of the word.
##    POS: The simple UPOS part-of-speech tag.
##    Tag: The detailed part-of-speech tag.
##    Dep: Syntactic dependency, i.e. the relation between tokens.
##    Shape: The word shape – capitalization, punctuation, digits.
##    is alpha: Is the token an alpha character?
##    is stop: Is the token part of a stop list, i.e. the most common words of the language?
# 
##    text  lemma   pos  tag  dep  shape  is_alpha  is_stop
##    text can be accessed as token.text etc.
## -----------------------------------------------------------------------------
## 
## Command line arguments:
## 1) -ipfile           : location for input file with the stt transcriptions
## 2) -opfileallposinfo : output file with parts-of-speech tagging (POS) for ALL non-stop words in sentences
## 3) -opfilekeyelem    : output file with only nouns in the required data structure that can be used
##                               to query the neo4j database later
## -----------------------------------------------------------------------------
## Usage example:
##    python3 id_elements_3.py -ipfile "/location/of/file.txt" -opfileallposinfo "/location/of/file/all_non_stop_Words_pos_info.txt" -opfilekeyelem "/location/of/file/only_nouns.txt"
## -----------------------------------------------------------------------------

import argparse
import os
import spacy
import copy
import numpy as np
#from spacy.lang.en.stop_words import STOP_WORDS
import json

import tkinter as tk
from functools import partial

argparser = argparse.ArgumentParser(
    description='parse the transcriptions, identify key elements and write output file')

argparser.add_argument(
    '-ipfile',
    '--inputfile',
    help='location for input file with the stt transcriptions')

argparser.add_argument(
    '-opfileallposinfo',
    '--opfileallwordsposinfo',
    help='location for output file where the key elements will be stored')

argparser.add_argument(
    '-opfilekeyelem',
    '--opfilekeyelementsonly',
    help='location for output file containing key elements')

def display_full_keywords_info(_key_elements_list):
    '''
    Display sentence-wise, the keywords information.
    VARIABLES PASSED:
          1) list of lists containing the sentence wise key words.
    RETURNS:
          nothing
    '''
    for idx1, sentence_key_words_list in enumerate(_key_elements_list):
        print(f"\tSentence {idx1+1} :")
        for idx2, key_word in enumerate(sentence_key_words_list):
            print(f"\t\tWord {idx2+1} : {key_word}")
    return

def display_keywords_info_particular_line(_key_elements_list, _line_no):
    '''
    Display sentence-wise, the keywords information.
    VARIABLES PASSED:
          1) list of lists containing the sentence wise key words.
          2) line number to display - corresponds to index of inner list
    RETURNS:
          nothing
    '''
    print(f"\n\tKeywords in sentence {_line_no+1} :")
    for idx, key_word in enumerate(_key_elements_list[_line_no]):
        print(f"\t\tWord {idx+1} : {key_word}")
    return

def sanity_check_keywords_list_validity(_key_elements_list):
    '''
    Perform santiy check.
    1) Outer list must contain exactly 2 or 3 elements.
    2) Each inner list to consist of 1 to 3 elements.
    VARIABLES PASSED:
          1) list of lists containing the sentence wise key words.
    RETURNS:
          boolean indicating if any issues found
            value         error situation
            True          no problems
            False         outer list did not contain exacty 2/3 elements
            False         some inner list did not contain exactly 1/2/3 elements
    '''
    if len(_key_elements_list) not in [2, 3]:
        print(f"\n\nFAILED sanity check of Keywords list : Expected a list with exactly two/ three elements.\n")
        return False
    for inner_list in _key_elements_list:
        if len(inner_list) not in [1, 2, 3]:
            print(f"\n\nFAILED sanity check of Keywords list : Some inner list did not contain exactly one/two/three elements.\n")
            return False
    return True ## all ok

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
    if _DEBUG_SWITCH:
        ## show keywords before any changes
        print(f"\n\nCandidate key words BEFORE any changes for sentence {_sentence_num} :")
        print(f"{_each_sentence_candidate_keywords}")
        print(f"\n\n")
    
    ## create and show the root window
    o_idkeyelem_root_window = c_idkeyelem_root_window(_DEBUG_SWITCH, _sentence_num, _each_sentence_candidate_keywords, _index_positions_deselected)
    o_idkeyelem_root_window.root.mainloop()

def id_elements_functionality(_DEBUG_SWITCH, args):
    ## setup spacy
    nlp = spacy.load('en_core_web_lg')
    POS_INTEREST_INCLUDE_VERBS_SWITCH = False
    #DEBUG_SWITCH = False

    ## process command line arguments
    ipfile            = args.inputfile              # -ipfile parameter
    opfileallposinfo  = args.opfileallwordsposinfo  # -opfileallposinfo parameter
    opfilekeyelem     = args.opfilekeyelementsonly  # -opfilekeyelem parameter
    
    ## check valid input file
    if not os.path.isfile(ipfile):
        print(f"FATAL ERROR: Input for ipfile parameter is not an existing file.\nExiting with RC=100")
        exit(100)
    else:
        ## read the file, storing each line as an element
        try:
            sentences_orig = [] # sentences to process
            with open(ipfile, 'r') as infile:
                for line in infile:
                    ## remove the new line and the period if present
                    sentences_orig.append(line.rstrip("\n").rstrip("."))
        except Exception as ipfile_read_error:
            print(f"\n\nFATAL ERROR: Problem reading the input file.\nError message: {ipfile_read_error}\nExiting with RC=200")
            exit(200)
        
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
    
    print(f"\n\nThe following sentences were read from the input file:\n")
    for idx, each_input_sentence in enumerate(sentences_orig):
        print(f"\tSentence {idx+1} :\n{each_input_sentence}")
    
    print(f"\n\nEach input sentence broken into words:\n")
    for idx, words_list in enumerate(sentences_words_list):
        print(f"\tSentence {idx+1} :\n{words_list}")
    
    print(f"\n\nAfter removing any stop words:\n")
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
    
    ### DEBUG DEBUG DEBUG - start
    # if DEBUG_SWITCH:
    #     print(f"\n\nAll non-stop words pos info:\n")
    #     for idx1, each_new_sent in enumerate(pos_info):
    #         print(f".........")
    #         for idx2, each_word_pos_info in enumerate(each_new_sent):
    #             print(f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}")
    ### DEBUG DEBUG DEBUG - end

    ## write the file for all non-stop words pos info
    try:
        with open(opfileallposinfo, 'w') as opfileall:
                json.dump(pos_info, opfileall)
    except Exception as opfileallpos_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file for all words pos info.\nError message: {opfileallpos_write_error}\nExiting with RC=500")
        exit(500)
    
    ## extract only the text of the noun and verb type words from the pos
    ##     info data structure then write it to output file
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
    
    ## write the final keywords data structure to file
    try:
        with open(opfilekeyelem, 'w') as opfilekeyonly:
                json.dump(final_keywords, opfilekeyonly)
    except Exception as opfilekeyonly_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file for key elements.\nError message: {opfilekeyonly_write_error}\nExiting with RC=520")
        exit(520)

if __name__ == '__main__':
    DEBUG_SWITCH = True
    args = argparser.parse_args()
    id_elements_functionality(DEBUG_SWITCH, args)
    print(f"\n\nNormal exit from program.\n")