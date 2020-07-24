## -----------------------------------------------------------------------------
## Goal:  Allow user to DESELECT images returned from the Neo4j query stage.
###       This is because some images may have wrongly found the objects of interest.
###       Each Neo4j query can return up to 20 images. But the maximum number of images per query to
###            pass to the next stage (Auto caption block) is limited to 5. So allow culling.
###       Logic:
###           1) Show the up to 20 images in a grid pattern as thumbnails.
###           2) Provide option to enlarge image to study more clearly.
###           3) Provide option to toggle the selection button.
###           4) Once selections confirmed, make sure number of remaining images is within maximum limit
###                   that is currently 5.
###       Outputs:
###          None
###              Only prints the before and after data structure to show that the images Deselected are
###                   are actually removed.
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
## Command line arguments: NONE
## -----------------------------------------------------------------------------
## Usage example:
##    python3 gui_query_neo_results_image_selection_1.py
## -----------------------------------------------------------------------------

# importing required packages
import tkinter as tk
from PIL import ImageTk, Image
import os
from functools import partial

class c_my_root_window:
    def __init__(self, _query_num, _each_query_result, _func_show_grid_selection_window):
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                    ----------------------------------",
            f"\n                    -----      INSTRUCTIONS      -----",
            f"\n                    ----------------------------------",
            f"\n",
            f"\nThis is the main control window for a particular query.",
            f"\n",
            f"\n\t1. Click on the button to proceed.",
            f"\n\t2. A grid displays for the candidate images from database query.",
            f"\n\t3. You can Deselect any images you want by toggling the Select button.",
            f"\n\t4. You can also Deselect all images.",
            f"\n\t5. But you can only Select maximum 5 images.",
            f"\n\t6. Once ready, click the Done button.",
            f"\n\t7. You can Enlarge any image antime.",
            f"\n",
            f"\n\t8. Important: If you select more than 5 images and press the Done button,",
            f"\n\t              you will have to start the whole process of selection again.",
            f"\n\t9. Important: If you accidentally close this window, the next query selection",
            f"\n\t              will start off.",
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
            width=110,
            height=20,
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
    def __init__(self, _root, _query_num, _root_done_button_deselected_positions_results, _num_candidate_images_in_this_query, _lbl_root_error_message):
        self.wnd = tk.Toplevel(master=_root)
        self.root = _root
        self.query_num = _query_num
        self.frame_arr = None
        self.frame_arr_n_rows = None
        self.frame_arr_n_cols = None
        self.max_limit_images_remaining_after_deselections = 5  ## after user makes deselections, the number of candidate images cannot be more than this limit
        self._num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        self.lbl_root_error_message = _lbl_root_error_message
        self._root_done_button_deselected_positions_results = _root_done_button_deselected_positions_results
        self.frm_done = tk.Frame(
            master=self.wnd,
            relief=tk.RAISED,
            borderwidth=10)
        self.btn_done = tk.Button(master=self.frm_done, text="Click to Confirm Deselections",
            bg="yellow", fg="black",
            command=self.do_button_done_processing
            )
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
            self.lbl_root_error_message["text"] = f"ERROR: After Deselections, number of images remaining = {num_images_remaining_after_deselections},\nwhich is greater than the maximum limit = {self.max_limit_images_remaining_after_deselections}\nYou need to restart the selection process...."
            self.lbl_root_error_message.configure(bg="red", fg="black", height=8, width=80)
            self.lbl_root_error_message.pack(padx=20, pady=20)
            self.wnd.destroy()
        else:
            ## all good so proceed to return information and destroy the root window
            ## complicated way to return values
            self._root_done_button_deselected_positions_results[0] = index_positions_of_deselected_result
            self.root.destroy()

    def build_frames(self, _n_rows, _n_cols, _in_img_arr_resized):
        ## the grid will always accomodate 20 images. The incoming array has tuple of the resized image and the
        ##     absolute path to image. But if the incoming array has less images, these entries will be None.

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
                if (r_idx % 3) == 0: ## picture frame
                    temp_array.append(my_frm_image(self.wnd, frame_name_str))
                elif (r_idx % 3) == 1: ## enlarge frame
                    temp_array.append(my_frm_enlarge(self.wnd, frame_name_str))
                else: ## select frame
                    temp_array.append(my_frm_select(self.wnd, frame_name_str))
            self.frame_arr.append(temp_array)
        
        ## set the frame arrays objects as required
        img_idx = 0
        for r_idx in range(_n_rows):
            for c_idx in range(_n_cols):
                if (r_idx) % 3 == 0: ## picture frame
                    self.frame_arr[r_idx][c_idx].frm_img.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    ## populate the image
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
                            bg="yellow", fg="red",
                            width=12, height=4
                            )
                        self.frame_arr[r_idx][c_idx].lbl_img.pack(padx=5, pady=5)
                elif (r_idx % 3) == 1: ## enlarge frame
                    self.frame_arr[r_idx][c_idx].frm_enlarge.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    self.frame_arr[r_idx][c_idx].enlarge_this_image = self.frame_arr[r_idx - 1][c_idx].image_with_path ## bcoz immediately previous row should reference the image frame
                    ## disable the Enlarge button if there is no associated image
                    if self.frame_arr[r_idx][c_idx].enlarge_this_image is None:
                        self.frame_arr[r_idx][c_idx].btn_enlarge.configure(state=tk.DISABLED)
                    self.frame_arr[r_idx][c_idx].btn_enlarge.pack(padx=5, pady=5)
                else: ## select frame
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
        self.wnd.columnconfigure(r_idx, weight=1, minsize=50)
        self.wnd.rowconfigure(r_idx, weight=1, minsize=50)
        self.frm_done.grid(row=r_idx, column=c_idx, rowspan=1, columnspan=_n_cols, sticky="nsew", padx=5, pady=5)
        self.btn_done.pack(padx=10, pady=10, fill='both', expand=True)

class my_frm_image:
    def __init__(self, _wnd, _frame_name):
        self.frm_img = tk.Frame(
            master=_wnd,
            relief=tk.SUNKEN,
            borderwidth=2)
        self.frame_name = _frame_name
        self.image_with_path = None
        self.lbl_img = tk.Label(master=self.frm_img, image=None)
        
    def add_image(self, _in_img):
        self.lbl_img.configure(image=_in_img)
        self.lbl_img.pack(padx=1, pady=1)

class my_frm_enlarge:
    def __init__(self, _wnd, _frame_name):
        self.frm_enlarge = tk.Frame(
            master=_wnd,
            relief=tk.RAISED,
            borderwidth=4)
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
    
    def do_enlarge_btn_press_functionality(self):
        #print(f"Would have enlarged image: {self.enlarge_this_image}")
        img_orig = ImageTk.PhotoImage(Image.open(self.enlarge_this_image))
        wnd_enlarged_img = tk.Toplevel(master=self.master)
        wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.enlarge_this_image)}")
        frm_enlarged_img = tk.Frame(
            master=wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        lbl_enlarged_img = tk.Label(
            master=wnd_enlarged_img,
            image=img_orig)
        lbl_enlarged_img.pack(padx=1, pady=1)
        frm_enlarged_img.pack()
        wnd_enlarged_img.mainloop()

class my_frm_select:
    def __init__(self, _wnd, _frame_name):
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
        if self.btn_select["text"] == 'Selected':
            ## Sink button and change text to Deselect
            self.btn_select.configure(text=f"Deselected", relief=tk.SUNKEN, bg="yellow", fg="black")
        else:
            ## Raise button and change text to Select
           self.btn_select.configure(text=f"Selected", relief=tk.RAISED, bg="black", fg="white")
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
    ##     Each row of image has associate Enlarge row and Select row i.e. three rows 
    ##          are logically are part of each image row.
    #      So the selection window grid will have have 2 x 3 x 10 = 60 frames.
    n_rows = 2 * 3
    n_cols = 10
    #print(f"n_rows = {n_rows}\t\tn_cols = {n_cols}\n")

    o_main_window = c_my_wnd_main_window(_root, _query_num, _root_done_button_deselected_positions_results, num_candidate_images_in_this_query, _lbl_root_error_message)
    o_main_window.build_frames(n_rows, n_cols, opened_imgs_arr)
    o_main_window.wnd.mainloop()

def select_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result):
    
    ## make a root window and show it
    o_root_window = c_my_root_window(_query_num, _each_query_result, show_grid_selection_window)
    #_query_num, _each_query_result, _func_show_grid_selection_window
    o_root_window.root.mainloop()
    o_root_window.done_button_deselected_positions_results
    if _DEBUG_SWITCH:
        print(f"\nQuery {_query_num} :::  o_root_window.done_button_deselected_positions_results = {o_root_window.done_button_deselected_positions_results}")
    return o_root_window.done_button_deselected_positions_results[0]
    
    #### ----------------------------------------------------------------------
    ####        OLDER CODE COMMENTED - start
    #### ----------------------------------------------------------------------
    ## make a root window and show it
    """
    msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
    msg_instructions_for_root = "".join([
        f"                    ----------------------------------",
        f"\n                    -----      INSTRUCTIONS      -----",
        f"\n                    ----------------------------------",
        f"\n",
        f"\nThis is the main control window for a particular query.",
        f"\n",
        f"\n\t1. Click on the button to proceed.",
        f"\n\t2. A grid displays for the candidate images from database query.",
        f"\n\t3. You can Deselect any images you want by toggling the Select button.",
        f"\n\t4. You can also Deselect all images.",
        f"\n\t5. But you can only Select maximum 5 images.",
        f"\n\t6. Once ready, click the Done button.",
        f"\n\t7. You can Enlarge any image antime.",
        f"\n",
        f"\n\t8. Important: If you select more than 5 images and press the Done button,",
        f"\n\t              you will have to start the whole process of selection again.",
        f"\n\t9. Important: If you accidentally close this window, the next query selection",
        f"\n\t              will start off.",
    ])
    root = tk.Tk()
    root.done_button_deselected_positions_results = [[]] ## eventually the inner list should be replaced by an interget list of positions
    root.title(F"Root Window - Query number {_query_num}")
    lbl_root_msg_warning_not_close = tk.Label(
        master=root,
        text=msg_warning_for_root_do_not_close,
        bg="red", fg="white",
        width=(len(msg_warning_for_root_do_not_close) + 10),
        height=5
        )
    lbl_root_instructions = tk.Label(
        master=root,
        text=msg_instructions_for_root,
        bg="blue", fg="white",
        justify=tk.LEFT,
        width=110,
        height=20,
        relief=tk.SUNKEN
        )
    lbl_root_error_message = tk.Label(
        master=root,
        text="No Errors detected so far",
        bg="blue", fg="white",
        justify=tk.LEFT,
        relief=tk.FLAT
        )
    btn_root_click_proceed = tk.Button(
        master=root,
        text=f"Click to proceed to selection for Query {_query_num}",
        bg="black", fg="white",
        relief=tk.RAISED,
        command=partial(show_grid_selection_window, root, _each_query_result, _query_num, root.done_button_deselected_positions_results) )
    lbl_root_instructions.pack(padx=10, pady=10)
    lbl_root_error_message.pack(padx=10, pady=10)
    lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
    btn_root_click_proceed.pack(padx=15, pady=15)
    root.mainloop()
    if _DEBUG_SWITCH:
        print(f"\nQuery {_query_num} ::: root.done_button_deselected_positions_results = {root.done_button_deselected_positions_results}")
    return root.done_button_deselected_positions_results[0]
    """
    #### ----------------------------------------------------------------------
    ####        OLDER CODE COMMENTED - start
    #### ----------------------------------------------------------------------

def selection_processing_functionality(_DEBUG_SWITCH):
    database_query_results = [
        [
            {'Image': '000000169542.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000169516.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000292186.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000168815.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000168743.jpg', 'Source': 'coco_test_2017'}
            ],
        [
            {'Image': '000000146747.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000509771.jpg', 'Source': 'coco_test_2017'}
            ],
        [
            {'Image': '000000012149.jpg', 'Source': 'coco_test_2017'}
            ]
        ]
    
    final_remaining_images_selected_info = []
    index_positions_to_remove_all_queries = []
    for query_num, each_query_result in enumerate(database_query_results):
        temp_array = []
        print(f"\n\nStarting Selection process for Query {query_num + 1}")
        num_candidate_images_before_selection_began = len(each_query_result)
        print(f"Number of images before selection began = {num_candidate_images_before_selection_began}\n")
        index_positions_to_remove_this_query = select_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        num_images_to_remove = len(index_positions_to_remove_this_query)
        print(f"\nNumber of images Deselected by user = {num_images_to_remove}.\nNumber of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }")
        ## remove the Deselected images
        for idx_each_image, each_image_info in enumerate(each_query_result):
            if idx_each_image not in index_positions_to_remove_this_query:
                temp_array.append(each_image_info)
        final_remaining_images_selected_info.append(temp_array)
        print(f"\nCompleted selection process - Query number {query_num + 1}\n")
    
    ## show summary info
    print(f"\n\n-------------------------------- SUMMARY INFORMATON --------------------------------")
    for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
        enumerate(zip(database_query_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
        print(f"For Query {query_num + 1}\nNumber of candidate images before selection = {len(each_query_results)}")
        print(f"Number of Deselections done = {len(each_query_index_positions_remove)}")
        print(f"Number of images remaining after Deselections = {len(each_query_final_remaining_images_info)}")
        if _DEBUG_SWITCH:
            print(f"\n\t------ Query images info BEFORE::\n{each_query_results}")
            print(f"\n\t------ Positions removed::\n{each_query_index_positions_remove}")
            print(f"\n\t------ Query images info AFTER::\n{each_query_final_remaining_images_info}\n\n")
    
    print(f"\n\nNormal exit.\n\n")

if __name__ == '__main__':
    DEBUG_SWITCH = True
    selection_processing_functionality(DEBUG_SWITCH)
    