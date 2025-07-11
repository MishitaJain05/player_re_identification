### Player Re-Identification Project

## Model Download

Manually add YOLO model to models folder with name "yolov11.pt" since the file is greater tham 100MB, it is not able to push on github

Also add all the three videos in videos folder

## Set Up

# 1. Clone the Repository and Navigate:

git clone https://github.com/MishitaJain05/player_re_identification.git

cd Player_Identification

# Dependencies - for manual setup - for python 3.10 or higher

pip install torch torchvision torchaudio opencv-python ultralytics

## Run the code

python src/crosscam.py

## Output

Output file in videos folder with name "output.mp4"
