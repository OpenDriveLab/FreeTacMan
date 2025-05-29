ros_dir="/opt/ros/noetic/"
if [ -d "$ros_dir" ]; then
    echo "$ros_dir : exists."
else
    echo "$ros_dir : does not exist."
    echo "please install ros"
    exit 1
fi
;
sudo apt update

echo "1. install miniconda3."

ros_dir="$(HOME)/miniconda3"
if [ -d "$ros_dir" ]; then
    echo "$ros_dir : exists."
else
    echo "start install miniconda3."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    $(HOME)/miniconda3/bin/conda init bash
    echo "install miniconda3 completed."
fi



echo "2. install ros_astra_camera"
sudo apt install libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev -y
sudo apt install ros-$ROS_DISTRO-image-geometry ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-publisher ros-$ROS_DISTRO-libuvc-ros 

ros_dir="$HOME/astra_ws/src/ros_astra_camera"
if [ -d "$ros_dir" ]; then
    echo "$ros_dir : exists."
else
    echo "start downloads ros_astra_camera."
    git clone https://github.com/orbbec/ros_astra_camera.git astra_ws/src/ros_astra_camera
fi

cd $HOME/astra_ws && catkin_make

source devel/setup.bash && rospack list && roscd astra_camera

./scripts/create_udev_rules
sudo udevadm control --reload && sudo  udevadm trigger

cd $HOME/astra_ws

res=$(grep -c "source $(pwd)/devel/setup.bash" ~/.bashrc)
if  [ $res -eq 0 ]; then
    echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
fi

echo "ros_astra_camera install completed"
cd $HOME

echo "2. install remote_control"

ros_dir="$HOME/remote_control/"
if [ -d "$ros_dir" ]; then
    echo "$ros_dir : exists."
else
    echo "start downloads remote_control."
    # git clone 
fi

sudo apt install can-utils net-tools libkdl-parser-dev -y

cd $HOME/remote_control

./tools/remove_build_file.sh && ./tools/build.sh

pip install opencv-python==4.7.0.72 matplotlib==3.7.1 h5py==3.8.0 dm-env==1.6 rosbag==1.16.0 catkin-pkg==1.0.0 empy==3.3.4