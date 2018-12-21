import numpy as np
import cv2
import os
import pickle


def calculate_descriptors(in_dir, t, bw, bh):
    descriptors = []
    src = "hw6_data/" + in_dir
    #Iterate through either test/training directory
    for root, dirs, files in os.walk(src):
        #Iterate through each background directory
        for folder in dirs:
            next_dir =  src + "/" + folder
            print(folder)
            #Iterate through each file
            for file in os.listdir(next_dir):
                
                #Read image and get its dimensions
                img = cv2.imread(os.path.join(next_dir, file))
                M, N = img.shape[0], img.shape[1]
                
                #Calculate deltas for width and height
                delta_w = int(M/(bw+1))
                delta_h = int(N/(bh+1))
                
                #Define our initial start and end for each block
                x_start, y_start = 0, 0
                x_end = int((2*delta_w))
                y_end = int((2*delta_h))
                
                #Here is a boolean that I use to initialize the start np array
                first = True
                
                #Iterate through each block, from range 0..bh and 0..bw
                for m in range(0, bh):
                    y_start = 0
                    for n in range(0, bw):
                        
                        #Splice the block and reshape
                        pixels = img[x_start:x_end, y_start:y_end] 
                        pixels = pixels.reshape(2*delta_h*2*delta_w, 3)
                        
                        #Calculate the histogram and unravel
                        hist, _ = np.histogramdd(pixels, (t,t,t))
                        hist_ravel = np.ravel(hist)
                        if(first == True):
                            descriptor = hist_ravel
                            first = False
                        else:
                            descriptor = np.append(descriptor, hist_ravel)
                            
                        #Move forward the value of the height delta for both start and end
                        y_start += delta_h
                        y_end += delta_h
                        
                    #Move forward the value of the width delta for both start and end
                    x_start += delta_w
                    x_end += delta_w
                    
                    #Reset the end value for
                    y_end = int(2*delta_h)
                    
                #Add the descriptor to a list
                descriptors.append(descriptor)    

    #Serialize our descriptors
    PIK = "p1_" + in_dir + ".dat"
    
    with open(PIK, "wb") as f:
        pickle.dump(descriptors, f)
    with open(PIK, "rb") as f:
        print(pickle.load(f))

if __name__ == "__main__":

    np.set_printoptions(suppress=True, threshold=np.nan)
    in_dir = "hw6_data/train"
    np.set_printoptions(suppress=True)
    t = 4
    bh = 4
    bw = 4

    calculate_descriptors("train", t, bw, bh)
    
    calculate_descriptors("test", t, bw, bh)
