ws_path=$(pwd)
echo $ws_path

cd $ws_path/camera_ws

file="$ws_path/camera_ws/devel/lib/astra_camera/list_devices_node"
if [[ -f "$file" ]]; then
    echo "list_devices_node exist."
else
    echo "list_devices_node no exist, please catkin_make astra_camera package.."
    exit 0
fi


file="/etc/udev/rules.d/56-orbbec-usb.rules"

if [[ -f "$file" ]]; then
    echo "camera-usb.rules exist."
else
    echo "camera-usb.rules not exist, please add camera-usb.rules."
    exit 0
fi


echo ""
echo "camera serial number: "
$ws_path/camera_ws/devel/lib/astra_camera/list_devices_node

echo "-----------------"
echo ""