#pragma once

#include <fcntl.h>
#include <chrono>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "buffers.hpp"
#include "utils.hpp"
#include "preprocessing.hpp"

namespace py = pybind11;

struct CameraFMT{
    v4l2_buf_type type;
    v4l2_field field;
    double fps;
    uint pixelformat;
    uint width; 
    uint height;
    std::string description;
};

inline std::ostream& operator<<(std::ostream& os, const CameraFMT& fmt){
    os << "Camera(width= " << fmt.width << ", height= " << fmt.height << ", fps= " << fmt.fps << ", format= " << fmt.description << ")";
    return os;
}

class CameraFMTList{
    private:
        CameraFMT* data;
        size_t capacity;
        size_t size;
        ssize_t best_idx;

    public:
        
        CameraFMTList() : data(nullptr), best_idx(-1), capacity(0), size(0){}
        ~CameraFMTList() {
            delete[] data;
        }
        
        void append(const CameraFMT& fmt, bool is_best = false){
            if (!data){
                capacity = 1;
                data = new CameraFMT[capacity];

            } else if (size == capacity){
                capacity*=2;
                CameraFMT* new_data = new CameraFMT[capacity];

                for (size_t i = 0; i < size; i++) {
                    new_data[i] = data[i];
                }
                delete[] data;
                data = new_data;
            }

            data[size] = fmt;
            if (is_best){
                best_idx = size;
            }
            size++;
        }

        std::string get_str() {
            std::ostringstream oss;
            for (int i = 0; i < size; i++){
                CameraFMT& fmt = data[i];
                oss << "Camera(" << "id= " << i << ", width= " << fmt.width << ", height= " << fmt.height << ", fps= " << fmt.fps << ", format= " << fmt.description;
                if (i == best_idx){
                    oss << ", BEST";
                }
                oss << ")\n";
            }
            return oss.str();
        }

        const CameraFMT& get_best_fmt() const {
            if (best_idx == -1){
                throw std::out_of_range("No best format");
            }

            return data[best_idx];
        }

        ssize_t get_best_index() const {
            return best_idx;
        }

        const CameraFMT& operator[](int index) const {
            if (index < 0 || index > size - 1){
                throw std::out_of_range("CameraFMTList index out of range");
            }

            return data[index];
        }
};

class Camera{
    private:
        int fd;
        ssize_t fmt_idx;
        CameraFMTList list;
        Preprocessor ppc;
    
        struct Stats{
            uint64_t n_frames;
            long long dequeue_ns;
            long long copy_ns;
            long long queue_ns;
            long long preprocess_ns;

            Stats() : n_frames(0), dequeue_ns(0), copy_ns(0), queue_ns(0), preprocess_ns(0) {}
        };

        Stats stats;

    protected:
        CameraRingBuffer h_ring;
        GPURingBuffer d_ring;
        bool is_streaming; 


    public: 
        Camera (const char* filename) : fd(open_camera(filename)), h_ring(fd), d_ring(h_ring.get_n_buffers()), fmt_idx(-1), is_streaming(false){
            check_capabilities();
            fetch_formats();
            set_best_format();
        }

        ~Camera(){
            try {
                stop_streaming();
            } catch(...) {}

            if(fd >= 0){
                close(fd);
            }
        }

        Camera(const Camera&) = delete;
        Camera& operator=(const Camera& other) = delete;

        int open_camera(const char* filename){
            int fd = open(filename, O_RDWR);
            if (fd == -1){
                std::ostringstream ss; ss << "Failed to open camera: " << strerror(errno);
                logging::error(ss);
                throw std::runtime_error("Failed to open camera device");
            }

            return fd;
        }

        void close_camera(){
            stop_streaming();

            if(fd >= 0){
                close(fd);
            }
        }

        friend class FrameGenerator;

        void check_capabilities(){
            v4l2_capability capabilities;
            clear(&capabilities);

            if(ioctl(fd, VIDIOC_QUERYCAP, &capabilities) == -1){
                std::ostringstream ss; ss << "Failed to get camera capabilities: " << strerror(errno);
                logging::error(ss);
                throw std::runtime_error("VIDIOC_QUERYCAP failed");
            } 

            std::ostringstream ss; ss << "Using camera: " << capabilities.card << " | Bus: " << capabilities.bus_info;
            logging::info(ss);

            if (check_for_flag(capabilities.device_caps, V4L2_CAP_STREAMING)){
                logging::info("Camera supports streaming");
            } else{
                logging::error("Camera does NOT support streaming");
                throw std::runtime_error("Camera does not support streaming");
            }

            if(check_for_flag(capabilities.device_caps, V4L2_CAP_EXT_PIX_FORMAT)){
                logging::info("Camera supports pixformat");
            } else{
                logging::error("Camera does NOT support pixformat");
                throw std::runtime_error("Camera does not support extended pixel formats");
            }

        }

        void fetch_formats(){
            double max_score = 0;
            for(int desc_idx= 0; ;desc_idx++){
                v4l2_fmtdesc desc;
                clear(&desc);
                desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; //single planar
                desc.index = desc_idx;

                if (ioctl(fd, VIDIOC_ENUM_FMT, &desc) == -1) {
                    if (errno == EINVAL) break;
                    std::ostringstream ss; ss << "Failed to enumerate formats: " << strerror(errno);
                    logging::error(ss);
                    break;
                }
                
                if (fmt_is_uncompressed(desc)){
                    for (int res_idx= 0;; res_idx++){
                        v4l2_frmsizeenum res;
                        clear(&res);
                        res.pixel_format = desc.pixelformat;
                        res.index = res_idx;

                        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &res) == -1) {
                            if (errno == EINVAL) {
                                break;
                            }
                            std::ostringstream ss; ss << "Failed to enumerate frame sizes: " << strerror(errno);
                            logging::error(ss);
                            break;
                        }

                        if (frm_is_discrete(res)){
                            for (int ival_idx = 0; ; ival_idx++) {
                                v4l2_frmivalenum ival;
                                clear(&ival);
                                ival.pixel_format = desc.pixelformat;
                                ival.index = ival_idx;
                                ival.width = res.discrete.width;
                                ival.height = res.discrete.height;
                                
                                if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == -1) {
                                    if (errno == EINVAL) {
                                        break;
                                    }
                                    std::ostringstream ss; ss << "Failed to enumerate frame intervals: " << strerror(errno);
                                    logging::error(ss);
                                    break;
                                }
                                
                                if (frm_ival_is_discrete(ival)) {
                                    double fps = (double)ival.discrete.denominator / ival.discrete.numerator;
                                    double score = fmt_score(fps, res.discrete.width, res.discrete.height);
                                    bool is_best = false;
                                    
                                    CameraFMT fmt = {V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_FIELD_NONE, fps, desc.pixelformat, res.discrete.width, res.discrete.height, std::string((char*)desc.description)};
                                    
                                    if (score > max_score){
                                        max_score = score;
                                        is_best = true;
                                    }

                                    list.append(fmt, is_best);
                                }
                            }
                        }
                    }
                }
            }
        }

        void list_formats(){
            std::ostringstream ss; ss << "Available formats:\n" << list.get_str();
            logging::info(ss);
        }

        void set_format(const int index){
            const CameraFMT& curr_fmt = list[index];
            
            v4l2_format format;
            clear(&format);
            
            format.type = curr_fmt.type;
            format.fmt.pix.width = curr_fmt.width;
            format.fmt.pix.height = curr_fmt.height;
            format.fmt.pix.pixelformat = curr_fmt.pixelformat;
            format.fmt.pix.field = curr_fmt.field;
            
            if(ioctl(fd, VIDIOC_S_FMT, &format) == -1){
                std::ostringstream ss; ss << "Failed to set format: " << strerror(errno);
                logging::error(ss);
            }else{
                fmt_idx = index;
                std::ostringstream ss; ss << "Camera format set: " << curr_fmt;
                logging::info(ss);
            }
        }

        void set_best_format(){
            int index = list.get_best_index();
            set_format(index);
        }

        void print_format() const {
            if (fmt_idx == -1) {
                throw std::runtime_error("No format to be retrieved as camera format has not been set");
            }

            const CameraFMT& fmt = list[fmt_idx];
            std::ostringstream ss; ss << "Current format: " << fmt;
            logging::info(ss);
        }

        void start_streaming(){
            if (is_streaming){
                return;
            }

            h_ring.start_streaming();
            d_ring.init(h_ring);
            is_streaming = true;
        }

        void stop_streaming(){
            if (!is_streaming){
                return;
            }

            h_ring.stop_streaming();
            is_streaming = false;
        }

        void reset_stats(){
            stats = Stats();
        }

        Stats get_stats() const{
            return stats;
        }

        py::dict get_stats_pydict() const{
            py::dict d;
            const double inv_1e6 = 1.0 / 1.0e6;

            auto mean_ms = [&](long long total_ns) -> double {
                if (stats.n_frames == 0){
                    return 0.0;
                }
                return (static_cast<double>(total_ns) * inv_1e6) / static_cast<double>(stats.n_frames);
            };

            d["n_frames"] = stats.n_frames;
            d["dequeue_mean_ms"] = mean_ms(stats.dequeue_ns);
            d["copy_mean_ms"] = mean_ms(stats.copy_ns);
            d["queue_mean_ms"] = mean_ms(stats.queue_ns);
            d["preprocess_mean_ms"] = mean_ms(stats.preprocess_ns);
            return d;
        }

        void accumulate_stats(
            const std::chrono::steady_clock::time_point& t0,
            const std::chrono::steady_clock::time_point& t1,
            const std::chrono::steady_clock::time_point& t2,
            const std::chrono::steady_clock::time_point& t3
        ){
            auto dequeue_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            auto copy_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            auto queue_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

            stats.n_frames += 1;
            stats.dequeue_ns += dequeue_ns;
            stats.copy_ns += copy_ns;
            stats.queue_ns += queue_ns;
        }

        void accumulate_preprocess(
            const std::chrono::steady_clock::time_point& t0,
            const std::chrono::steady_clock::time_point& t1
        ){
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            stats.preprocess_ns += ns;
        }

        torch::Tensor preprocess(int idx){
            const CameraFMT& fmt = list[fmt_idx];
            
            auto t0 = std::chrono::steady_clock::now();
            ppc.process(d_ring[idx], fmt.height, fmt.width);
            auto t1 = std::chrono::steady_clock::now();
            accumulate_preprocess(t0, t1);

            torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
            return torch::from_blob(d_ring[idx].out, {3, (long)fmt.height, (long)fmt.width}, opts);
        }
};



class FrameGenerator {
    private:
        Camera* cam;

    public:
        FrameGenerator() : cam(nullptr) {}
        FrameGenerator(Camera* _cam ) : cam(_cam) {}

        FrameGenerator* __iter__(){
            return this;
        }

        torch::Tensor __next__(){
            if(!cam->is_streaming){
                cam->start_streaming();
            }

            auto t0 = std::chrono::steady_clock::now();
            int idx = cam->h_ring.dequeue_buffer();
            auto t1 = std::chrono::steady_clock::now();

            if (idx == -1){
                throw py::stop_iteration();
            }

            auto t2_start = std::chrono::steady_clock::now();
            cam->d_ring.copy_buffer_to_gpu(idx, cam->h_ring.buffer_start(idx));
            auto t2 = std::chrono::steady_clock::now();

            cam->h_ring.queue_buffer(idx);
            auto t3 = std::chrono::steady_clock::now();

            cam->accumulate_stats(t0, t1, t2, t3);
            
            return cam->preprocess(idx);
        }
};
