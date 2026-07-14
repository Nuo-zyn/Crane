/*
 * Nuo.h
 * 用法:
 *   Nuonuo zyn;
 *   zyn.open("/dev/ttyUSB0", 115200);
 *   zyn.sendDigits("235");      // 发送数字
 *   zyn.sendBeans("YG");        // 发送豆子
 */

#ifndef SERIAL_SENDER_H
#define SERIAL_SENDER_H

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cerrno>

class Nuonuo {
public:
    static constexpr uint8_t FRAME_HEAD = 0xAA;
    static constexpr uint8_t FRAME_END = 0xBB;
    static constexpr uint8_t TYPE_DIGIT  = 0x01;
    static constexpr uint8_t TYPE_BEAN   = 0x02;

    Nuonuo() : fd_(-1) {}
    ~Nuonuo() { close(); }

    bool open(const std::string& port, int baudrate = 115200) {
        fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) return false;

        struct termios tty;
        tcgetattr(fd_, &tty);

        speed_t spd = B115200;
        if (baudrate == 9600) spd = B9600;
        else if (baudrate == 38400) spd = B38400;
        else if (baudrate == 57600) spd = B57600;

        cfsetospeed(&tty, spd);
        cfsetispeed(&tty, spd);
        tty.c_cflag = CS8 | CLOCAL | CREAD;
        tty.c_iflag = 0;
        tty.c_oflag = 0;
        tty.c_lflag = 0;
        tcsetattr(fd_, TCSANOW, &tty);
        return true;
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool isOpen() const { return fd_ >= 0; }

    bool write(const uint8_t* data, size_t len) {
        if (!isOpen()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        while (total < len) {
            ssize_t n = ::write(fd_, data + total, len - total);
            if (n > 0) {
                total += n;
            } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                return false;
            }
        }
        return true;
    }

    void sendDigits(const std::string& digits) {
        std::vector<uint8_t> frame;
        frame.reserve(2 + digits.size() + 1);
        frame.push_back(FRAME_HEAD);
        frame.push_back(TYPE_DIGIT);
        for (char c : digits) frame.push_back(static_cast<uint8_t>(c));
        frame.push_back(FRAME_END);
        write(frame.data(), frame.size());
    }

    void sendBeans(const std::string& beans) {
        std::vector<uint8_t> frame;
        frame.reserve(2 + beans.size() + 1);
        frame.push_back(FRAME_HEAD);
        frame.push_back(TYPE_BEAN);
        for (char c : beans) frame.push_back(static_cast<uint8_t>(c));
        frame.push_back(FRAME_END);
        write(frame.data(), frame.size());
    }

private:
    int fd_;
    std::mutex mutex_;
};

#endif