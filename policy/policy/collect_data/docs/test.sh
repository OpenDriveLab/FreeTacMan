
echo "install ros please input 1 or 0"
read input1

ros_dir="/opt/ros/noetic/"
if [ -d "$ros_dir" ]; then
    input1=2
else
    echo
fi

case $input1 in
    1)
        echo "Start install ros..."
        wget http://fishros.com/install -O fishros && . fishros
        ;;
    2)  
        echo "ROS was already installed before."
        ;;
    *)
        echo "Do not install ROS."
        # exit 0 # 退出脚本并返回成功代码 0
        ;; 
esac


echo ""