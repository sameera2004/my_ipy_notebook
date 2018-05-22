from __future__ import print_function
import os

i = 1

for f_name in os.listdir("."):
    if f_name.startswith("con_0005"):
          print (f_name)
          os.rename(f_name, "con_0005_" + str(i) + ".nii")
          i = i + 1
