# software implementing Face Recognition using python  

# ******* libraries & modules *******

import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import face_recognition as fr


# ******* getting details into python using pandas *******

f=pd.read_csv('Employee.csv')
# print(f.to_string())
emp_no=f["Employee No"].tolist()
name=f["Name"].tolist()
filename=f["Photo Location"].tolist()
n= len(emp_no)
emp=[]
emp_encode=[]


#  ******* unknown pic that to be recognise *******

uk=fr.load_image_file("uk.jpeg",'RGB')
uk_encode=fr.face_encodings(uk)[0]
# print("bill gate pic shape :  ",uk.shape)  #it will give shape(pix,pix,3)


#  ******* loading our predata pictures for matching ******* 

for i in range(n):
    emp.append(fr.load_image_file(filename[i],'RGB'))
    emp_encode.append(fr.face_encodings(emp[i])[0])


#  ******* comparing unknown pic with predata pictures *******
 
found= fr.compare_faces(emp_encode,uk_encode,tolerance=0.5)
# print(found) # it will return either true/false

# ******* face location and rectangle corners *******
l=fr.face_locations(uk)
top=l[0][0]
right=l[0][1]
bottom=l[0][2]
left=l[0][3]
  
#font type..
fnt=ImageFont.truetype("fonts/arial",40)  

# ******* after detection of face, writing his name on it and showing 

for i in range(n):
    if found[i]:
        print(f"face matches with picture {i+1}")
        left=100
        bottom=emp[i].shape[0]
        pil_uk=Image.fromarray(uk)
        draw=ImageDraw.Draw(pil_uk)
        draw.text((left,bottom-50),name[i],font=fnt,fill=(0,255,0))
        draw.rectangle(
                (left,top,right,bottom),
                outline=(0,0,255),width=4)
        pil_uk.show()


# output: a image will be dispayed with his name written on it.

# ******* THANKYOU *******