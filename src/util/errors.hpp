#pragma once

#include <string>
#include <exception>

class ScannerError: public std::exception {
private:
    std::string _msg;
public:
    ScannerError(const std::string& msg) : _msg(msg) {}
    virtual const char* what() const noexcept override {
        return _msg.c_str();
    }
};

class EncoderInitError: public std::exception {
private:
    std::string _msg;
public:
    EncoderInitError(const std::string& msg) : _msg(msg) {}
    virtual const char* what() const noexcept override {
        return _msg.c_str();
    }
};