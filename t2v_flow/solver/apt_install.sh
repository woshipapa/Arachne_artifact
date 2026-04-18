#!/bin/bash


conf_yum() {
    local url=$1
      rm -rf /etc/yum.repos.d/* &&   wget -O /etc/yum.repos.d/Centos-base.repo $url
      dnf clean all  
      dnf update
      dnf makecache
}



conf_apt() {
    local url=$1

      rm /etc/apt/sources.list 
      wget -O /etc/apt/sources.list $url

    # apt-key add /root/ubuntu.gpg
    # cp -r /root/trusted.gpg.d/ /etc/apt/
      apt-get update 
      apt-get upgrade  

}

check_system_conf() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        
        case $ID in
            ubuntu)
                case $VERSION_ID in            
                    18.04)
                        print_color success "Ubuntu system detected."
                        conf_apt download.chinatelecom.ai/Bionic-ChinaTelecom.list
                        ;;
                    20.04)
                        print_color success "Ubuntu system detected."
                        conf_apt download.chinatelecom.ai/Focal-ChinaTelecom.list
                        ;;
                    22.04)
                        print_color success "Ubuntu system detected."
                        conf_apt download.chinatelecom.ai/Jammy-ChinaTelecom.list
                        ;;
                    24.04)
                        print_color success "Ubuntu system detected."
                        conf_apt download.chinatelecom.ai/Noble-ChinaTelecom.list
                        ;;                   
                esac
                ;;            

            centos)
                case $VERSION_ID in
                    7)
                        print_color success "CentOS 7 system detected."
                        conf_yum download.chinatelecom.ai/CentOS7-ChinaTelecom.repo
                        ;;
                    8)
                        print_color success "CentOS 8 system detected."
                        conf_yum download.chinatelecom.ai/CentOS8-ChinaTelecom.repo
                        ;;
                    *)
                        print_color error "Unsupported CentOS version: $VERSION_ID"
                        exit 1
                        ;;
                esac
                ;;
            ctyunos)
                case $VERSION_ID in
                    2.0.1)
                        print_color success "ctyunos 2.0.1 system detected."
                        conf_yum download.chinatelecom.ai/CTyunOS-2.0.1-ChinaTelecom.repo
                        ;;
                    22.06)
                        print_color success "ctyunos 22.06 system detected."
                        conf_yum download.chinatelecom.ai/CTyunOS-22.06-ChinaTelecom.repo
                        ;;
                    23.01)
                        print_color success "ctyunos 23.06 system detected."
                        conf_yum  download.chinatelecom.ai/CTyunOS-23.01-ChinaTelecom.repo
                        ;;
                    *)
                        print_color error "Unsupported CentOS version: $VERSION_ID"
                        exit 4
                        ;;
                esac
                ;;
            *)
                print_color error "Unsupported system: $PRETTY_NAME"
                exit 1
                ;;
        esac
    else
        print_color error "System information file not found."
        exit 1
    fi
}

print_color() {
    local type=$1
    local message=$2
    case $type in
        success)
            echo -e "\033[1;32m$message\033[0m" # 绿色字体
            ;;
        error)
            echo -e "\033[1;31m$message\033[0m" # 红色字体
            ;;
        info)
            echo -e "\033[1;33m$message\033[0m" # 黄色字体
            ;;
        *)
            echo "$message"
            ;;
    esac
}


main() {
    check_system_conf
}


main