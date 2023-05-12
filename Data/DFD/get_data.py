import os
current_address = os.path.dirname(os.path.abspath('/mnt/shared/deepfake/FF++/faces/original_sequences/actors/c23/01__exit_phone_room.mp4'))
file_list = os.listdir(current_address)
with open("DFD_Real.txt","w") as f:
    for file_address in file_list:
        file_address = os.path.join(current_address, file_address)

        for filename in os.listdir(file_address):
            print(file_address + '/' + filename)
            
            f.write('0,'+ file_address + '/' + filename + '\n') 


    # if os.path.isfile(file_address):
    #     print("这个是文件，文件名称：", file_address)
    # elif os.path.isdir(file_address):
    #     print("这个是文件夹，文件夹名称：", file_address)
    # else:
    #     print("这个情况没遇到")