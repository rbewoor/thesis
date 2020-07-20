## -----------------------------------------------------------------------------
###             FUNCTIONAL programming approach
### Goal: Do STT processing on the input wav files, then do NLP processing to extract the keywords,
###          and then query the Neo4j database to return matching images.
## 
##  1) STT part:
##     Use Deepspeech model to run inference on the input wav files.
##     Outputs:
##        a) Intermediate data:
##           for use by the identify keywords logic
## 
##  2) Identify keywords
##        Process the input data which has the speech to text transcriptions.
##        Using Spacy, remove stop-words, perform part of speech tagging.
##        Write all POS info for these non stopwords to a file (with type of noun or verb).
##        Get user to drop one or more words from the keywords identified.
##        Then perform sanity check on the remaning keywords to ensure the data structure 
##           satisfies requirements for the neo4j query module.
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
##         ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
##          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
##          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
##          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
##          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
##          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
##          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
##          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
##          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
##          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -wavlocfile       : location of input file with each line specifying the individual wav files to be processed.
## 2) -opfileallposinfo : location where to write a file with parts-of-speech (POS) info.
## -----------------------------------------------------------------------------
## Usage example:
##    python3 combined_stt_id_query_1.py -wavlocfile "/home/rohit/PyWDUbuntu/thesis/SttTranscribe/stt_wav_files_loc_1.txt" -opfileallposinfo "/location/of/file/all_non_stop_Words_pos_info.txt"
## -----------------------------------------------------------------------------

## import necessary packages
##   imports of common modules
import argparse
import os
import json
##   imports for stt logic
import subprocess
##   imports for identify keywords logic
import spacy
import copy
import numpy as np
#from spacy.lang.en.stop_words import STOP_WORDS - not required as using alternative approach
##   imports for neo4j query logic
import sys
from py2neo import Graph

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

def change_candidate_elements(_key_elements_list):
    '''
    ACTIONS:
        Shows the keywords list for each sentence before any changes are made.
        Accepts user selection for the word positions to be dropped. Validates the values entered.
        Drops the words specified by user selection if allowed to.
        Calls the sanity check function to ensure the remaining words meet the requirements for
              query module in downstream processing.
    ACCEPTS:
        list of lists containing the sentence wise key words.
    RETURN:
        1) changes_made_flag : A boolean value indicating if any changes made
                False if no changes required and/ or made by user
                True  if any changes are made
        2) the keywords list (changed or unchanged as per user selection)
    '''
    changes_made_flag = False ## set as true at start
    good_to_go_flag = False   ## set as true at start
    ## show the list as is at start
    print(f"\n\nCANDIDATE key words before any changes:")
    display_full_keywords_info(_key_elements_list)
    print(f"\n\n")
    
    drop_message = f"\n\tEnter the position number for each word that should be DROPPED."                 + \
        f"\n\tTo drop no words, simply enter 0 without any spaces after that."                            + \
        f"\n\tTo drop more than one word, enter multiple numbers separated by a space."                   + \
        f"\n\tE.g. to drop words in positions one, two and five; you must enter:\n1 2 5"                  + \
        f"\n\tTo drop exactly one word, you must enter only that number without any spaces"               + \
        f"\n\tE.g. To drop only the second word; you must enter:\n2"
    while not good_to_go_flag:
        ## accept user input, convert to integers. Test all values are valid. If any problem then don't process and ask to reenter input.
        try:
            tempcopy_key_elements_list = copy.deepcopy(_key_elements_list)
            for idx, words_list in enumerate(_key_elements_list):
                print(f"\n{drop_message}")
                display_keywords_info_particular_line(_key_elements_list, idx)
                print(f"\nPlease input the words to drop:")
                user_selection_list = [int(val) for val in input().split(' ')]
                user_selection_list.sort()
                if user_selection_list[0] == 0:
                    ## no changes desired to the keywords list for this sentence
                    invalid_user_selection_flag = False
                    continue
                elif [True for val in user_selection_list if val > len(_key_elements_list[idx])]:
                    print(f"\nERROR: Some selection value or values were outside valid range. All values should have been less than {len(_key_elements_list[idx])}.\n")
                    good_to_go_flag = False
                    invalid_user_selection_flag = True
                    break
                elif len(user_selection_list) >= len(_key_elements_list[idx]):
                    print(f"\nERROR: Number of selections cannot be more than the number of words already present. Dropping ALL the keywords is not allowed.\n")
                    good_to_go_flag = False
                    invalid_user_selection_flag = True
                    break
                ## reaching here means, all selection values are valid. Not all the words will be dropped. Now process dropping of those words from the keywords list
                np_user_selection_list = np.array((user_selection_list))
                for new_idx in range(len(np_user_selection_list)):
                    ### DEBUG - start
                    print(f"DROPPED this word at position {user_selection_list[new_idx]}: {tempcopy_key_elements_list[idx].pop(np_user_selection_list[new_idx] - 1)}")
                    ### DEBUG - end
                    np_user_selection_list -= 1
                    changes_made_flag = True
                invalid_user_selection_flag = False
            ## perform sanity check on remaining values in the keywords list if applicable
            if not invalid_user_selection_flag:
                good_to_go_flag = sanity_check_keywords_list_validity(tempcopy_key_elements_list)
                if not good_to_go_flag:
                    print(f"\n\nSANITY CHECK FAILED. You will need to re-enter selections.")
        except Exception as user_selection_error:
        #except ValueError:
            print(f"\n\nYour input selection had a problem. Start again from scratch. Enter numbers to indicate which keywords to drop. E.g. to drop words in position one, two and five, you must enter 1 2 5. To drop only second position, enter just 2")
    if changes_made_flag:
        _key_elements_list = copy.deepcopy(tempcopy_key_elements_list)

    return changes_made_flag, _key_elements_list

def id_elements_functionality(_stt_module_results, _opfileallposinfo, _DEBUG_SWITCH):
    '''
    High level module to execute the identify keywords functionality by processing the 
         transcriptions from the STT functionality.
    VARIABLES PASSED:
          1) Array containing the transcriptions
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
    
    ### DEBUG DEBUG DEBUG - start
    if _DEBUG_SWITCH:
        print(f"\n\nEach input sentence broken into words:\n")
        for idx, words_list in enumerate(sentences_words_list):
            print(f"\tSentence {idx+1} :\n{words_list}")
    ### DEBUG DEBUG DEBUG - end
    
    print(f"\n\nWords of each input sentence after removing any stop words:\n")
    for idx, words_list in enumerate(sentences_words_list_no_stop):
        print(f"\tSentence {idx+1} :\n{words_list}")
    
    ### DEBUG DEBUG DEBUG - start
    if _DEBUG_SWITCH:
        print(f"\n\nJoining the non-stop words as a new sentence (for readability only):\n")
        for idx, new_sent_no_stop in enumerate(sentences_no_stop):
            print(f"\tNew sentence {idx+1} :\n{new_sent_no_stop}")
    ### DEBUG DEBUG DEBUG - end
    
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
    if _DEBUG_SWITCH:
        print(f"\n\nAll non-stop words pos info:\n")
        for idx1, each_new_sent in enumerate(pos_info):
            print(f".........")
            for idx2, each_word_pos_info in enumerate(each_new_sent):
                print(f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}")
    ### DEBUG DEBUG DEBUG - end

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
    
    ## create a blank result array
    combined_key_elements = []
    for sent_info in pos_info:
        each_sentence_key_elements = []
        for each_word_info in sent_info:
            if each_word_info['tag_'] in pos_tags_of_interest:
                each_sentence_key_elements.append(each_word_info['lemma_'])
        if each_sentence_key_elements: # append to the final list only if its not an empty list
            combined_key_elements.append(each_sentence_key_elements)
    
    ## check the key words are within valid bounds, allow user to drop keywords, get final acceptance by the user
    changes_made_flag, combined_key_elements = change_candidate_elements(combined_key_elements)
    if changes_made_flag:
        print(f"\n\nFINAL keywords AFTER changes:")
    else:
        print(f"\n\nNO CHANGES MADE : Final key elements same as Candidate key elements.")
    display_full_keywords_info(combined_key_elements)
    
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - start
    ## any required changes are done, write the final key words data structure to file
    # try:
    #     with open(opfilekeyelem, 'w') as opfilekeyonly:
    #             json.dump(combined_key_elements, opfilekeyonly)
    # except Exception as opfilekeyonly_write_error:
    #     print(f"\n\nFATAL ERROR: Problem creating the output file for key elements.\nError message: {opfilekeyonly_write_error}\nExiting with RC=520")
    #     exit(520)
    # print(f"\n\nCOMPLETED ID KEYWORDS LOGIC.\n")
    ### COMMENTED FOR COMBINED PROGRAM LOGIC - end
    
    return (0, None, combined_key_elements)
    

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
## ----------------------  CONTROL LOGIC STARTS                    CONTROL LOGIC STARTS ---------------------- 
###############################################################################################################

def execute_control_logic(args):
    DEBUG_SWITCH = False
    neo4j_query_result_limit = 10

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
    
    #exit(0) ## DEBUG

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
  
    ## show the result if no problem in query
    print(f"\nDatabase query results:")
    if query_neo4j_logic_RC == 0:
        for idx, (v1, v2) in enumerate(zip(id_elements_module_results, query_neo4j_module_results)):
            print(f"\n{idx+1}) Keywords: {v1}\nQuery result:\n{v2}")
    else:
        print(f"\n\nFATAL ERROR: Database query logic failed. Return code = {query_neo4j_logic_RC}\n\tMessage = {query_neo4j_logic_msg}")
        print(f"\n\nPremature Exit.\n\n")
        exit(300)
    
    print(f"\n\n\nNormal exit from program.\n")
    return

if __name__ == '__main__':
    args = argparser.parse_args()
    execute_control_logic(args)
    exit(0)