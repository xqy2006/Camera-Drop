#pragma once

#include "config.hpp"

#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

enum class Scene {Encoder, Decoder};

class ArgParser {
public:
    ArgParser(const int argc, const char** argv, Scene scene) : is_valid_(true), should_exit_(false), scene_(scene) {
        if(scene == Scene::Encoder) parse_encoder_args(argc, argv);
        else if(scene == Scene::Decoder) parse_decoder_args(argc, argv);
        else set_error("Unknown scene");
    }

    bool is_valid() const {return is_valid_;}
    bool should_exit() const {return should_exit_;}
    const std::string& error_message() const {return error_msg_;}

private:
    bool is_valid_;
    bool should_exit_;
    Scene scene_;
    std::string error_msg_;

    void set_error(const std::string& error_msg){
        is_valid_ = false;
        error_msg_ = error_msg;
    }

    bool parse_integer(const std::string& str, int& out_val) const {
        try{
            size_t pos = 0;
            out_val = std::stoi(str, &pos, 10);
            return pos == str.length();
        } catch(...){
            return false;
        }
    }

    bool parse_float(const std::string& str, float& out_val) const {
        try{
            size_t pos = 0;
            out_val = std::stof(str, &pos);
            return pos == str.length();
        } catch(...){
            return false;
        }
    }

    void parse_encoder_args(const int argc, const char** argv){
        for(*argv++; *argv and is_valid_; ++argv){
            std::string arg = *argv;

            if(arg == "-i" or arg == "--input"){
                if(*(argv + 1)) Config::INPUT_VIDEO_FILE = *(++argv);
                else set_error("Missing value for " + arg);
            }


            else if(arg == "-o" or arg == "--output"){
                if(*(argv + 1)) Config::OUTPUT_VIDEO_FILE = *(++argv);
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-z" or arg == "--zstd"){
                if(*(argv + 1)){
                    int val;
                    if(parse_integer(*(++argv), val)) Config::COMPRESSION_LEVEL = val;
                    else set_error("Invalid integer value for " + arg);
                }
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-r" or arg == "--redundancy"){
                if(*(argv + 1)){
                    float val;
                    if(parse_float(*(++argv), val)) Config::REDUNDANCY_FACTOR = val;
                    else set_error("Invalid float value for " + arg);
                }
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-f" or arg == "--fps"){
                if(*(argv + 1)){
                    int val;
                    if(parse_integer(*(++argv), val)) Config::OUTPUT_FPS = val;
                    else set_error("Invalid integer value for " + arg);
                }
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-h" or arg == "--help"){
                print_encoder_help();
                should_exit_ = true;
                return;
            }

            else set_error("Unknown argument: " + arg);
        }

        if(is_valid_ and !should_exit_) validate();
    }

    void parse_decoder_args(const int argc, const char** argv){
        for(argv++; *argv and is_valid_; ++argv){
            std::string arg = *argv;

            if(arg == "-i" or arg == "--input"){
                if(*(argv + 1)) Config::INPUT_VIDEO_FILE = *(++argv);
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-v" or arg == "--vout"){
                if(*(argv + 1)) Config::VOUT_FILE = *(++argv);
                else set_error("Missing value for " + arg);
            }

            else if(arg == "-h" or arg == "--help"){
                print_decoder_help();
                should_exit_ = true;
                return;
            }

            else set_error("Unknown argument: " + arg);
        }
        if(is_valid_ and !should_exit_) validate();
    }

    void print_encoder_help() const {
        puts("Usage: encoder -i <input> [options]");
        puts("Options:");
        puts("  -i, --input <file>     Input file path (Required)");
        puts("  -o, --output <file>    Output file path (Default: output.mp4)");
        puts("  -z, --zstd <level>     Zstd compression level 1-22 (Default: 9)");
        puts("  -r, --redundancy <f>   Fountain redundancy factor 1.0~20.0 (Default: 2.0)");
        puts("  -f, --fps <value>          Video FPS 1-60 (Default: 15)");
        puts("  -h, --help             Show this help message");
    }

    void print_decoder_help() const {
        puts("Usage: decoder -i <input> [options]");
        puts("Options:");
        puts("  -i, --input <file>     Input file path (Required)");
        puts("  -v, --vout <file>      Validity tag file (Default: vout.bin)");
        puts("  -h, --help             Show this help message");
    }

    void validate(){
        if(scene_ == Scene::Encoder){
            if(Config::INPUT_VIDEO_FILE.empty()){
                set_error("Input file (-i) is required.");
                return;
            }
            if(Config::COMPRESSION_LEVEL < 1 or Config::COMPRESSION_LEVEL > 22){
                set_error("Zstd compression level must be between 1 and 22.");
                return;
            }
            if(Config::REDUNDANCY_FACTOR < 1.0f or Config::REDUNDANCY_FACTOR > 20.0f){
                set_error("Redundancy factor must be between 1.0 and 20.0.");
                return;
            }
            if(Config::OUTPUT_FPS < 1 or Config::OUTPUT_FPS > 60){
                set_error("Video FPS must be between 1 and 60.");
                return;
            }
        }

        if(scene_ == Scene::Decoder){
            if(Config::INPUT_VIDEO_FILE.empty()){
                set_error("Input file (-i) is required.");
                return;
            }
        }
    }
};