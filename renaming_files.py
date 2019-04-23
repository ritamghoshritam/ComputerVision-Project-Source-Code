import os

path = os.getcwd()

#path to the data folder
data_path = 'C:\\Users\\Dhiraj\\Desktop\\imagepreprocessing\\images'
# C:\Users\Dhiraj\Desktop\project\dataset'

data_dir = 'data'

data_list = os.listdir(os.path.join(data_path,data_dir))

os.chdir(os.path.join(data_path,data_dir))

#the base name of the image files

base_name='tennis'

for i in range(len(data_list)):
    img_name = data_list[i]
    img_rename = base_name + '_{:06d}'.format(i+1)+'.jpg'
    if not os.path.exists(img_rename):
        os.rename(img_name,img_rename)

data_dir = 'tennis'

data_list = os.listdir(os.path.join(data_path,data_dir))

os.chdir(os.path.join(data_path,data_dir))
 
for i in range(len(data_list)):
    img_name = data_list[i]
    img_rename = base_name + '_{:06d}'.format(i+1)+'.jpg'
    if not os.path.exists(img_rename):
        os.rename(img_name,img_rename)
       
os.chdir(path)