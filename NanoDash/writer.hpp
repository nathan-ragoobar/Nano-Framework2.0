#pragma once

#include <string>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <condition_variable>

class MetricWriter {
private:
    struct MetricData {
        std::string name;
        double value;
        std::chrono::system_clock::time_point timestamp;
    };

    std::string filename_;
    std::thread writer_thread_;
    std::queue<MetricData> write_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool running_;
    std::map<std::string, size_t> metric_positions_;

    void writerLoop() {
        std::ofstream file(filename_, std::ios::app);
        
        // Write CSV header if file is empty
        if (file.tellp() == 0) {
            file << "Timestamp,";
            // Write metric names as headers
            size_t max_pos = 0;
            for (const auto& metric : metric_positions_) {
                max_pos = std::max(max_pos, metric.second);
            }
            std::vector<std::string> headers(max_pos + 1);
            for (const auto& metric : metric_positions_) {
                headers[metric.second] = metric.first;
            }
            for (size_t i = 0; i < headers.size(); ++i) {
                file << headers[i];
                if (i < headers.size() - 1) file << ",";
            }
            file << "\n";
        }
        
        while (running_ || !write_queue_.empty()) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            if (write_queue_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::seconds(1), 
                    [this]() { return !write_queue_.empty() || !running_; });
                continue;
            }

            MetricData data = write_queue_.front();
            write_queue_.pop();
            lock.unlock();

            // Format timestamp
            auto time_t = std::chrono::system_clock::to_time_t(data.timestamp);
            std::tm tm = *std::localtime(&time_t);
            std::stringstream timestamp_ss;
            timestamp_ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");

            // Create array of placeholder values
            std::vector<std::string> values(metric_positions_.size(), "");
            values[metric_positions_[data.name]] = stringifyValue(data.value);

            // Write CSV line
            file << timestamp_ss.str();
            for (const auto& value : values) {
                file << "," << (value.empty() ? "NA" : value);
            }
            file << "\n";
            file.flush();
        }
    }

    std::string stringifyValue(double value) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(6) << value;
        return ss.str();
    }

public:
    MetricWriter(const std::string& filename, 
                const std::vector<std::string>& metric_names) 
        : filename_(filename + ".csv"), running_(true) {  // Add .csv extension
        
        // Initialize metric positions
        for (size_t i = 0; i < metric_names.size(); ++i) {
            metric_positions_[metric_names[i]] = i;
        }

        // Start writer thread
        writer_thread_ = std::thread(&MetricWriter::writerLoop, this);
    }

    ~MetricWriter() {
        close();
    }

    void addScalar(const std::string& name, double value) {
        if (metric_positions_.find(name) == metric_positions_.end()) {
            throw std::runtime_error("Unknown metric name: " + name);
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        write_queue_.push({
            name,
            value,
            std::chrono::system_clock::now()
        });
        lock.unlock();
        queue_cv_.notify_one();
    }

    void close() {
        if (running_) {
            running_ = false;
            queue_cv_.notify_all();
            if (writer_thread_.joinable()) {
                writer_thread_.join();
            }
        }
    }
};