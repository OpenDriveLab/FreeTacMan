
#!/bin/bash
architecture=$(uname -m)
echo $architecture
case $architecture in
    x86_64)
        echo "System architecture is x86"
        echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list
        echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
        echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list
        echo "deb http://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list
        ;;
    arm*)
        echo "System architecture is ARM"
        echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal main restricted universe multiverse" > /etc/apt/sources.list
        echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list
        echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list
        echo "deb http://ports.ubuntu.com/ubuntu-ports/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list
        ;;
    *)
        echo "Unknown architecture: $architecture"
        exit 1 # 结束脚本，并返回错误代码 1
        ;;
esac


sudo apt update

sudo apt install build-essential vim cmake git mlocate net-tools openssh-server pkg-config -y

sudo updatedb

echo "install ros please input 1 or 0"
read option

case $option in
    1)
        echo "Start install ros..."
        wget http://fishros.com/install -O fishros && . fishros
        ;;
    *)
        echo "Exiting script."
        exit 0 # 退出脚本并返回成功代码 0
        ;;
esac

ros_dir="/opt/ros/noetic/"
if [ -d "$ros_dir" ]; then
    echo "Directory $ros_dir exists."
else
    echo "Directory $ros_dir does not exist."
fi


