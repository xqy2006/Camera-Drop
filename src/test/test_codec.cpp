#include "codec/Encoder.hpp"
#include "codec/Decoder.hpp"
#include "util/config.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cassert>
#include <algorithm>

std::vector<uint8_t> generate_data(size_t size){
    std::vector<uint8_t> data(size);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
    std::generate(data.begin(), data.end(), 
        [&](){
            return static_cast<uint8_t>(dist(rng));
        }
    );
    return data;
}

typedef std::vector<uint8_t> Packet;

class VideoChannel {
public:
    VideoChannel(const double lr = 0.0, const double er = 0.00) : loss_rate_(lr), error_rate_(er), total_packet_(0), lossed_packet_(0) {}
    void trans(const Packet& data){
        auto raw = data;
        ++total_packet_;
        
        static thread_local std::mt19937 rng(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        );

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        if(prob(rng) < loss_rate_){      // 模拟丢包
            ++lossed_packet_;
            return;
        }

        for(size_t i = 0; i < raw.size(); ++i){
            if(prob(rng) < error_rate_){ // 简单模拟 flip
                raw[i] = ~raw[i];
            }
        }
        
        packets_.push_back(raw);
    }

    std::vector<Packet> recieved() const {
        return packets_;
    }

    int total_packet() const {return total_packet_;}
    int lossed_packet() const {return lossed_packet_;}

private:
    std::vector<Packet> packets_;
    int total_packet_;
    int lossed_packet_;
    double loss_rate_;
    double error_rate_;
};

int main(){

    puts("getting encoder...");

    auto data = generate_data(1024 * 1024 * 10); // 10 MB
  //  std::string str = "hello world.";
  //  std::vector<uint8_t> data(str.begin(), str.end());

    const char* inFile = "in.bin";
    const char* outFile = "out.bin";

    FILE* f = fopen(inFile, "wb");
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);

    Encoder encoder(inFile);
    assert(encoder.is_valid());

    puts("Ready to send.");

    VideoChannel channel(0.0, 0.05);

  //  const uint32_t packet_count = encoder.packet_count_recommended();
    const uint32_t packet_count = std::max(encoder.packet_count_recommended(), 10u);

    for(uint32_t i = 0; i < packet_count; ++i){
        auto packet = encoder.get_packet();
      //  printf("packet size: %u\n", packet.size());
        channel.trans(packet);
    }

    auto recieved = channel.recieved();

    Decoder decoder;

    int cnt = 0;

    for(auto& packet : recieved){
        bool res = decoder.process_packet(packet);
        if(res) ++cnt;
        if(decoder.is_complete()){
            puts("Decode complete!");
            decoder.save_to_file(outFile);
            break;
        }
    }
    if(!decoder.is_complete()) puts("Decode failed.");
    printf("sent: %d, loss: %d, decoded: %d\n", channel.total_packet(), channel.lossed_packet(), cnt); 
}
