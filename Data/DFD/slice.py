file_image_dir = "/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/Data/DFD/DFD_Real.txt"
fh = open(file_image_dir, 'r')
lines = fh.readlines()
train_real = lines[0:int(len(lines)*0.7)]
val_real = lines[int(len(lines)*0.7):int(len(lines)*0.85)]
test_real = lines[int(len(lines)*0.85):int(len(lines))]
print(len(train_real))

file_image_dir = "/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/Data/DFD/DFD_Fake.txt"
fh = open(file_image_dir, 'r')
lines = fh.readlines()
train_fake = lines[0:int(len(lines)*0.7)]
val_fake = lines[int(len(lines)*0.7):int(len(lines)*0.85)]
test_fake = lines[int(len(lines)*0.85):int(len(lines))]

print(len(train_fake))


with open('/mnt/traffic/home/pankun/FF++/Deepfake-Detection-master/Data/DFD/test.txt','w') as f:
    for i in test_real:
        f.write(i)

    for i in test_fake:
        f.write(i)


