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
#include <iostream>

class MetricWriter {
private:
struct MetricData {
    std::string name;
    double value;
    std::chrono::system_clock::time_point timestamp;
    int64_t step;  // Add step number
};

    std::string filename_;
    std::thread writer_thread_;
    std::queue<MetricData> write_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool running_;
    std::map<std::string, size_t> metric_positions_;
    int64_t current_step_;

    void writerLoop() {
        // Open with both flags to create file if it doesn't exist
        std::ofstream file(filename_, std::ios::out | std::ios::app);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename_ << std::endl;
            running_ = false;
            return;
        }
        
        // Write CSV header if file is empty
        if (file.tellp() == 0) {
            file << "Timestamp,Step,"; // Add Step column explicitly

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
            file.flush();
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
            file << timestamp_ss.str() << "," << data.step;  // Add step to output
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
    : filename_(filename + ".csv"), running_(true), current_step_(0) {

    // Initialize metric positions
    // Ensure both training and validation loss are included
    bool has_train_loss = false;
    bool has_val_loss = false;

    for (size_t i = 0; i < metric_names.size(); ++i) {
        metric_positions_[metric_names[i]] = i;
        if (metric_names[i] == "train_loss") has_train_loss = true;
        if (metric_names[i] == "val_loss") has_val_loss = true;
    }

    // Add loss metric if not present
    if (!has_train_loss) {
        metric_positions_["train_loss"] = metric_positions_.size();
    }
    if (!has_val_loss) {
        metric_positions_["val_loss"] = metric_positions_.size();
    }

    writer_thread_ = std::thread(&MetricWriter::writerLoop, this);
    }

    ~MetricWriter() {
        close();
    }

    // Helper methods for training and validation loss
    void addTrainingLoss(double value, int64_t step = -1) {
        addScalar("train_loss", value, step);
    }

    void addValidationLoss(double value, int64_t step = -1) {
        addScalar("val_loss", value, step);
    }

    void addScalar(const std::string& name, double value, int64_t step = -1) {
        if (metric_positions_.find(name) == metric_positions_.end()) {
            throw std::runtime_error("Unknown metric name: " + name);
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (step == -1) {
            step = ++current_step_;
        }
        write_queue_.push({
            name,
            value,
            std::chrono::system_clock::now(),
            step
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