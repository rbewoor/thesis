## PARTIALLY WORKING - Button - with image resizing - all 20 images

#keep for refernce

## to change a button features later on e.g. 
##      button.configure(relief=tk.SUNKEN, state=tk.DISABLED)
##      button.configure(relief=tk.RAISED, state=tk.ACTIVE)    

##  when Select button is pressed, toggle the relief and the text

##  any Select button press causes the last button to toggle

# importing required packages
import tkinter as tk
from PIL import ImageTk, Image
import os

img_folder = r'/home/rohit/PyWDUbuntu/thesis/Imgs2Detect_20imgs/'

image_files_list = [os.path.join(img_folder, f) \
                    for f in os.listdir(img_folder) \
                    if os.path.isfile(os.path.join(img_folder, f))]
print(f"num of images = {len(image_files_list)}\narray=\n{image_files_list}\n")

print(f"num of images = {len(image_files_list)}")

tk_wnd = tk.Tk()

## selection functionality : select button is pressed
def do_select_btn_press_functionality():
    if tk_btn_select["text"] == 'Select':
        ## Sink button and change text to Deselect
        tk_btn_select.configure(text=f"Deselect", relief=tk.SUNKEN)
    else:
        ## Raise button and change text to Select
        tk_btn_select.configure(text=f"Select", relief=tk.RAISED)
    return

num_rows_of_images = 4

opened_imgs_arr = []
for img_idx in range(20):
    img_orig = Image.open(image_files_list[img_idx])
    img_resized = img_orig.resize((100, 80),Image.ANTIALIAS)
    opened_imgs_arr.append(ImageTk.PhotoImage(img_resized))
    del img_orig, img_resized

img_idx = 0
for ridx in range(3 * num_rows_of_images):
    tk_wnd.columnconfigure(ridx, weight=1, minsize=50)
    tk_wnd.rowconfigure(ridx, weight=1, minsize=50)
    
    #for cidx in range(4):
    for cidx in range(int(len(image_files_list) / num_rows_of_images)):
        tk_frm_img = tk.Frame(
            master=tk_wnd,
            relief=tk.SUNKEN,
            borderwidth=2)
        tk_frm_enlarge = tk.Frame(
            master=tk_wnd,
            relief=tk.RAISED,
            borderwidth=4)
        tk_frm_select = tk.Frame(
            master=tk_wnd,
            relief=tk.RAISED,
            borderwidth=4)
        ## setup appropriate frame
        if (ridx % 3) == 0: ## picture Label
            tk_frm_img.grid(row=ridx, column=cidx, padx=5, pady=5)
            ## read the image
            tk_lbl_img = tk.Label(master=tk_frm_img, image = opened_imgs_arr[img_idx])
            img_idx += 1
            tk_lbl_img.pack(padx=1, pady=1)
        elif (ridx % 3) == 1: ## enlarge Button
            tk_frm_enlarge.grid(row=ridx, column=cidx, padx=5, pady=5)
            tk_btn_enlarge = tk.Button(
                master=tk_frm_enlarge,
                text=f"Enlarge",
                bg="black", fg="white")
            tk_btn_enlarge.pack(padx=5, pady=5)
        else: ## (ridx % 3) must = 2 ## select Button
            tk_frm_select.grid(row=ridx, column=cidx, padx=5, pady=5)
            tk_btn_select = tk.Button(
                master=tk_frm_select, 
                text=f"Select", 
                bg="black", fg="white",
                command=do_select_btn_press_functionality
            )
            tk_btn_select.pack(padx=5, pady=5)

tk_wnd.mainloop()

print(f"\n\nNormal exit.\n\n")