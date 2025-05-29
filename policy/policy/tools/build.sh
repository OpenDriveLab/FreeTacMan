ws_path=$(pwd)
echo $ws_path

echo "build camera"
cd $ws_path/camera_ws
catkin_make
cd ..

echo "build remote_control"
cd $ws_path/remote_control
./tools/build.sh
cd ..

echo "print camera serial"
tools/camera_serial.sh