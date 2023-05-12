

file=open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/test.txt', 'r')
list_read = file.readlines()

Face2Face = []
Deepfakes = []
FaceSwap = []
NeuralTextures = []
for i in list_read:
    if 'Face2Face' in i:
        Face2Face.append(i)
    if 'Deepfakes' in i:
        Deepfakes.append(i)
    if 'FaceSwap' in i:
        FaceSwap.append(i)    
    if 'NeuralTextures' in i:
        NeuralTextures.append(i)

file= open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/Face2Face_test.txt', 'w')  
for fp in Face2Face:
    file.write(str(fp))
file.close()

file= open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/Deepfakes_test.txt', 'w')  
for fp in Deepfakes:
    file.write(str(fp))
file.close()

file= open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/FaceSwap_test.txt', 'w')  
for fp in FaceSwap:
    file.write(str(fp))
file.close()

file= open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/data_list/NeuralTextures_test.txt', 'w')  
for fp in NeuralTextures:
    file.write(str(fp))
file.close()
