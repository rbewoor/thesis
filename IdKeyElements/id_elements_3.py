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
    
    drop_message = f"\nType the number for each word that should be DROPPED."          + \
        f"\nTo drop no words, simply enter 0"                                          + \
        f"\nTo drop more than one word, enter multiple numbers separated by a space."  + \
        f"\nE.g. to drop words in position one, two and five, you must enter 1 2 5"    + \
        f"\nTo drop only second position, enter just 2\nYour input:\t"
    while not good_to_go_flag:
        ## accept user input, convert to integers. Test all values are valid. If any problem then don't process and ask to reenter input.
        try:
            tempcopy_key_elements_list = copy.deepcopy(_key_elements_list)
            for idx, words_list in enumerate(_key_elements_list):
                display_keywords_info_particular_line(_key_elements_list, idx)
                print(f"\n{drop_message}")
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

def id_elements_functionality(args):
    ## setup spacy
    nlp = spacy.load('en_core_web_lg')
    POS_INTEREST_INCLUDE_VERBS_SWITCH = False
    DEBUG_SWITCH = False

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
    
    ### DEBUG DEBUG DEBUG - start
    if DEBUG_SWITCH:
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
    
    ## extract the only the text of the noun and verb type words from the pos
    ##     info data structure then write it to output file
    ## tags of interest as per naming convention https://spacy.io/api/annotation#pos-tagging
    pos_tags_of_interest_nouns = ['NN', 'NNS']
    pos_tags_of_interest_verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    if POS_INTEREST_INCLUDE_VERBS_SWITCH:
        pos_tags_of_interest = pos_tags_of_interest_nouns + pos_tags_of_interest_verbs
    else:
        pos_tags_of_interest = pos_tags_of_interest_nouns
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
    
    ## any required changes are done, write the final key words data structure to file
    try:
        with open(opfilekeyelem, 'w') as opfilekeyonly:
                json.dump(combined_key_elements, opfilekeyonly)
    except Exception as opfilekeyonly_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file for key elements.\nError message: {opfilekeyonly_write_error}\nExiting with RC=520")
        exit(520)
    
    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    id_elements_functionality(args)
