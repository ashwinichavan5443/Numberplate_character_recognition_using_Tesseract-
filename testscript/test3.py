import glob
import argparse
import os

parser = argparse.ArgumentParser(description='argument for image for pytesseract')
parser.add_argument('-d', '--dir', type=str, required=True,
                    help='Path for the image to be pass')

args = parser.parse_args()
num=1
row=1
string = args.dir + "*.jpg"
#print(glob.glob(string))
for file in glob.glob(string):
	string2="python test2.py -r"+str(row)+" -n "+str(num)+" -i "+ file 
	print (string2)
	os.system(string2)
	
	row+=1