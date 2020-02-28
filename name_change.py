import os

def rename():
    i = 0
    path = r'data/train/groundtruth/'

    filelist = os.listdir(path)   #该文件夹下所有的文件（包括文件夹）
    for files in filelist:   #遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files)    #原来的文件路径
        if os.path.isdir(Olddir):       #如果是文件夹则跳过
                continue
        filename = ''     #文件名
        filetype = '.nii.gz'        #文件扩展名
        Newdir = os.path.join(path, filename + str(i) + filetype)   #新的文件路径
        os.rename(Olddir, Newdir)    #重命名
    return True

def rename_2(path):
    filelist = os.listdir(path)
    for files in filelist:
        olddir = os.path.join(path,files)
        filename = files.replace('Case','')
        filetype = '.nii.gz'
        newdir = os.path.join(path,filename)
        os.rename(olddir,newdir)
rename_2(r'data/train/image')