/*
 * @Author: zhoups_xx 1013700083@qq.com
 * @Date: 2023-05-11 14:39:14
 * @LastEditors: zhoups_xx 1013700083@qq.com
 * @LastEditTime: 2023-05-11 16:09:10
 * @FilePath: /list-device/list_device.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <librealsense2/rs.hpp>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

using namespace std;

void saveSerialToFile(vector<string> snum, std::string& file_path)
{
    if(!snum.empty())
    {
        ofstream ofs;
        string str;
    
        ofs.open(file_path,ios::trunc);

        std::cout << std::endl;
        
        str = "#! /bin/bash\n";

        for(int i =0; i < snum.size(); ++i)
        {   
            std::string content = "export camera" + std::to_string(i)+ "_serial_number=";
            std::string sn = snum[i];

            std::string ss = content + sn;

            if(i < snum.size()-1)
            {
                str += ss + "\n";
            }
            else
            {
                str += ss;
            }
        }
        std::cout << str << std::endl; 
        ofs << str << endl;
        ofs.close();
    }
}

vector<string> getRSSerialNum()
{
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    vector<string> list_sn;
    if (devices.size() == 0)
        {
            std::cerr << "No device connected, please connect a RealSense device" << std::endl;
            // ROS_ERROR("No device connected, please connect a RealSense device");

            //To help with the boilerplate code of waiting for a device to connect
            //The SDK provides the rs2::device_hub class
            rs2::device_hub device_hub(ctx);

            //Using the device_hub we can block the program until a device connects
            // selected_device = device_hub.wait_for_device();
            return list_sn;
        }
        else
        {
            std::cout << "Found the following devices:\n" << std::endl;

            // device_list is a "lazy" container of devices which allows
            //The device list provides 2 ways of iterating it
            //The first way is using an iterator (in this case hidden in the Range-based for loop)
//            int index = 0;
            for (rs2::device device : devices)
            {
                // std::cout << "  " << index++ << " : " << get_device_name(device) << std::endl;
                if(device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
                {
                    string sn = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                    list_sn.push_back(sn);
                    cout << "Serial number: " << sn << endl;
                }
            }
        }

        return list_sn;
}

int main(int argc, char ** argv)try
{
   
    string home = getenv("HOME");
    string file_path = home + "/.aloha_camera_config.bash";
    
    vector<string> list_sn;
    list_sn = getRSSerialNum();
    
    if(list_sn.empty())
    {   
        std::cout << "list SN is empty" << std::endl;
        return -1;
    }

    // 写入文件
    // saveSerialToFile(list_sn, file_path);
    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}