#import the library opencv
import cv2

#globbing utility.
import glob

#select the path
path = "C:\Users\irsya\Documents\Python\deskew-master\raw_images\*.*"
for file in glob.glob(path):
    print(file)
    g=+1

i = cv2.imread(file)
j= cv2.imwrite('ready2skew'+g+'.png',i)
#a= cv2.imread()
print(j)

# %%%%%%%%%%%%%%%%%%%%%
#conversion numpy array into rgb image to show
c = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
cv2.imshow('Color image', c)
#wait for 1 second
k = cv2.waitKey(1000)
#destroy the window
cv2.destroyAllWindows()