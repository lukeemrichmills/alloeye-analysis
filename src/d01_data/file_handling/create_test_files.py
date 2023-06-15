"""one off script to create truncated data files for testing scripts faster"""

import pandas as pd
import platform
from os import listdir

# define hard-coded variables
import pandas.errors

input_dir = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\"
output_dir = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\test\\"
line_truncation = 8000

# import list of file names
only_files = listdir(input_dir)
csv_files = []  # define empty list to populate
for f in only_files:  # for each item in directory...
    if f[-4:] == '.csv':  # filter by csv ...
        if 'GazeData' in f:
            headers = 'EyeTimestamp,UnityTime,EyeFrameSequence,FrameCount,FPS,Object,PosX,PosY,PosZ,ScaleX,'\
                      'ScaleY,ScaleZ,ColX,ColY,ColZ,Trial,ViewNo,ViewPos,ViewingAngle,Rot?,ObjShifted,MoveCode,'\
                      'cameraX,cameraY,cameraZ,camRotX,camRotY,camRotZ,leftPupilDiameter,rightPupilDiameter,'\
                      'leftEyeOpenness,rightEyeOpenness,RcontrollerX,RcontrollerY,RcontrollerZ,LcontrollerX,'\
                      'LcontrollerY,LcontrollerZ,RcontrRotX,RcontrRotY,RcontrRotZ,LcontrRotX,LcontrRotY,LcontrRotZ,'\
                      'lGazeOrigin_x,lGazeOrigin_y,lGazeOrigin_z,rGazeOrigin_x,rGazeOrigin_y,rGazeOrigin_z,'\
                      'lGazeDirection_x,lGazeDirection_y,lGazeDirection_z,rGazeDirection_x,rGazeDirection_y,'\
                      'rGazeDirection_z,gazeConvergenceDist,gazeConvDistValidity,Object_notable,posX_notable,'\
                      'posY_notable,posZ_notable,scaleX_notable,scaleY_notable,scaleZ_notable,colX_notable,'\
                      'colY_notable,colZ_notable'
            try:
                df = pd.read_csv(input_dir + "\\" + f, header=None, skiprows=1)
                df.columns = headers.split(',')
            except pandas.errors.EmptyDataError:
                df = pd.DataFrame(columns=headers.split(','))
                print(f"empty dataframe created for {f}")

        elif 'EventLog' in f:
            try:
                df = pd.read_csv(input_dir + "\\" + f, header=None, skiprows=1)
                df = df.iloc[:, 0:3]
                df.columns = "Unity time,frame,event".split(',')

            except:
                print(f"could not parse {f}")
        else:
            try:
                df = pd.read_csv(input_dir + "\\" + f)
            except:
                print(f'could not parse {f}')  # doesn't seem to work on ObjectPositions files

        if len(df.index) > line_truncation:
            df = df.iloc[:line_truncation]
        try:
            df.to_csv(output_dir + "\\" + f, index=False)
        except:
            print(f"could not write {f}")
