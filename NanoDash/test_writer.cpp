#include <gtest/gtest.h>
#include "writer.hpp"  // Assuming the class is in this header
#include <fstream>
#include <filesystem>
#include <thread>
#include <regex>
#include <chrono>
#include <random>

class MetricWriterTest : public ::testing::Test {
    protected:
        const std::string test_filename = "test_metrics";  // .csv will be added by MetricWriter
        std::vector<std::string> test_metrics{"loss", "accuracy", "perplexity"};
        
        void SetUp() override {
            // Ensure test file doesn't exist before each test
            if (std::filesystem::exists(test_filename + ".csv")) {
                std::filesystem::remove(test_filename + ".csv");
            }
        }
    
        // Helper function to read the contents of the test file
        std::vector<std::string> readTestFile() {
            std::vector<std::string> lines;
            std::ifstream file(test_filename + ".csv");
            std::string line;
            while (std::getline(file, line)) {
                lines.push_back(line);
            }
            return lines;
        }
    
        // Helper function to parse CSV line
        std::vector<std::string> parseCSVLine(const std::string& line) {
            std::vector<std::string> values;
            std::stringstream ss(line);
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                values.push_back(token);
            }
            return values;
        }
    
        bool isValidTimestamp(const std::string& timestamp) {
            std::regex timestamp_regex(R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})");
            return std::regex_match(timestamp, timestamp_regex);
        }
    };
    

// Test CSV header creation
TEST_F(MetricWriterTest, CSVHeaderCreation) {
    MetricWriter writer(test_filename, test_metrics);
    
    auto lines = readTestFile();
    ASSERT_EQ(lines.size(), 1);  // Should have header only
    
    auto header_columns = parseCSVLine(lines[0]);
    ASSERT_EQ(header_columns.size(), 4);  // Timestamp + 3 metrics
    ASSERT_EQ(header_columns[0], "Timestamp");
    ASSERT_EQ(header_columns[1], "loss");
    ASSERT_EQ(header_columns[2], "accuracy");
    ASSERT_EQ(header_columns[3], "perplexity");
}

// Test adding a single scalar value
TEST_F(MetricWriterTest, AddSingleScalar) {
    MetricWriter writer(test_filename, test_metrics);
    writer.addScalar("loss", 0.5);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto lines = readTestFile();
    ASSERT_EQ(lines.size(), 2);  // Header + 1 data line
    
    auto values = parseCSVLine(lines[1]);
    ASSERT_EQ(values.size(), 4);  // Timestamp + 3 metrics
    ASSERT_TRUE(isValidTimestamp(values[0]));
    ASSERT_EQ(values[1], "0.500000");  // loss value
    ASSERT_EQ(values[2], "NA");  // accuracy placeholder
    ASSERT_EQ(values[3], "NA");  // perplexity placeholder
}

// Test multiple values in same timestamp
TEST_F(MetricWriterTest, MultipleValuesInTimestamp) {
    MetricWriter writer(test_filename, test_metrics);
    
    writer.addScalar("loss", 0.5);
    writer.addScalar("accuracy", 0.75);
    writer.addScalar("perplexity", 2.5);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto lines = readTestFile();
    ASSERT_EQ(lines.size(), 4);  // Header + 3 data lines
    
    auto last_values = parseCSVLine(lines.back());
    ASSERT_EQ(last_values.size(), 4);
    ASSERT_TRUE(isValidTimestamp(last_values[0]));
    ASSERT_NE(last_values[1], "NA");
    ASSERT_NE(last_values[2], "NA");
    ASSERT_NE(last_values[3], "NA");
}

// Test timestamp format
TEST_F(MetricWriterTest, TimestampFormat) {
    MetricWriter writer(test_filename, test_metrics);
    writer.addScalar("loss", 0.5);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto lines = readTestFile();
    ASSERT_FALSE(lines.empty());
    
    std::string timestamp = lines[0].substr(0, 19);  // Extract timestamp portion
    ASSERT_TRUE(isValidTimestamp(timestamp));
}

// Test invalid metric name
TEST_F(MetricWriterTest, InvalidMetricName) {
    MetricWriter writer(test_filename, test_metrics);
    ASSERT_THROW(writer.addScalar("invalid_metric", 0.5), std::runtime_error);
}

// Test proper cleanup on destruction
TEST_F(MetricWriterTest, CleanupOnDestruction) {
    {
        MetricWriter writer(test_filename, test_metrics);
        writer.addScalar("loss", 0.5);
    }  // Writer goes out of scope here
    
    // File should still exist and be readable
    ASSERT_TRUE(std::filesystem::exists(test_filename + ".csv"));  // Add .csv extension
    auto lines = readTestFile();
    ASSERT_FALSE(lines.empty());
}

// Test concurrent writes
TEST_F(MetricWriterTest, ConcurrentWrites) {
    MetricWriter writer(test_filename, test_metrics);
    
    const int num_threads = 10;
    const int writes_per_thread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread([&writer, writes_per_thread]() {
            for (int j = 0; j < writes_per_thread; ++j) {
                writer.addScalar("loss", 0.5);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }));
    }
    
    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Give some time for final writes
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto lines = readTestFile();
    ASSERT_EQ(lines.size(), num_threads * writes_per_thread);
}

// Test file append mode
TEST_F(MetricWriterTest, FileAppendMode) {
    {
        MetricWriter writer1(test_filename, test_metrics);
        writer1.addScalar("loss", 0.5);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto initial_lines = readTestFile();
    
    {
        MetricWriter writer2(test_filename, test_metrics);
        writer2.addScalar("loss", 0.6);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto final_lines = readTestFile();
    
    ASSERT_EQ(final_lines.size(), initial_lines.size() + 1);
}
/*
// This test generates training-like data and leaves the file for visualization
TEST_F(MetricWriterTest, GenerateVisualizationData) {
    // Use a different file for this test to avoid cleanup
    const std::string viz_filename = "visualization_data.txt";
    
    // Define metrics we want to track
    std::vector<std::string> metrics = {
        "loss", "accuracy", "perplexity", "learning_rate", "tokens_per_second"
    };
    
    MetricWriter writer(viz_filename, metrics);
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initial values
    double loss = 5.0;
    double accuracy = 0.2;
    double perplexity = 10.0;
    double learning_rate = 0.001;
    double tokens_per_sec = 1000.0;
    
    // Generate 40 data points with realistic training patterns
    for (int i = 0; i < 40; ++i) {
        // Simulate training patterns:
        // - Loss decreases over time but with some noise
        loss *= std::uniform_real_distribution<>(0.95, 0.99)(gen);
        
        // - Accuracy increases but plateaus
        accuracy = std::min(0.95, accuracy + std::uniform_real_distribution<>(0.01, 0.02)(gen));
        
        // - Perplexity decreases with some fluctuation
        perplexity *= std::uniform_real_distribution<>(0.95, 0.98)(gen);
        
        // - Learning rate decays
        learning_rate *= std::uniform_real_distribution<>(0.98, 0.995)(gen);
        
        // - Tokens per second varies randomly
        tokens_per_sec += std::uniform_real_distribution<>(-50.0, 50.0)(gen);
        tokens_per_sec = std::max(800.0, std::min(1200.0, tokens_per_sec));

        // Write values
        writer.addScalar("loss", loss);
        writer.addScalar("accuracy", accuracy);
        writer.addScalar("perplexity", perplexity);
        writer.addScalar("learning_rate", learning_rate);
        writer.addScalar("tokens_per_second", tokens_per_sec);
        
        // Small delay to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Verify the file exists and has content
    ASSERT_TRUE(std::filesystem::exists(viz_filename));
    auto lines = readTestFile();
    ASSERT_EQ(lines.size(), 200);  // 40 iterations * 5 metrics
    
    std::cout << "\nGenerated visualization data in 'visualization_data.txt'\n";
    std::cout << "The file contains " << lines.size() << " data points.\n";
    std::cout << "You can now use this file with your visualization software.\n\n";
}
*/

// This test generates visualization data in CSV format
TEST_F(MetricWriterTest, GenerateVisualizationData) {
    const std::string viz_filename = "visualization_data";
    std::vector<std::string> metrics = {
        "loss", "accuracy", "perplexity", "learning_rate", "tokens_per_second"
    };
    
    MetricWriter writer(viz_filename, metrics);
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initial values
    double loss = 5.0;
    double accuracy = 0.2;
    double perplexity = 10.0;
    double learning_rate = 0.001;
    double tokens_per_sec = 1000.0;
    
    // Generate 40 data points with realistic training patterns
    for (int i = 0; i < 40; ++i) {
        // Simulate training patterns:
        // - Loss decreases over time but with some noise
        loss *= std::uniform_real_distribution<>(0.95, 0.99)(gen);
        
        // - Accuracy increases but plateaus
        accuracy = std::min(0.95, accuracy + std::uniform_real_distribution<>(0.01, 0.02)(gen));
        
        // - Perplexity decreases with some fluctuation
        perplexity *= std::uniform_real_distribution<>(0.95, 0.98)(gen);
        
        // - Learning rate decays
        learning_rate *= std::uniform_real_distribution<>(0.98, 0.995)(gen);
        
        // - Tokens per second varies randomly
        tokens_per_sec += std::uniform_real_distribution<>(-50.0, 50.0)(gen);
        tokens_per_sec = std::max(800.0, std::min(1200.0, tokens_per_sec));

        // Write values
        writer.addScalar("loss", loss);
        writer.addScalar("accuracy", accuracy);
        writer.addScalar("perplexity", perplexity);
        writer.addScalar("learning_rate", learning_rate);
        writer.addScalar("tokens_per_second", tokens_per_sec);
        
        // Small delay to simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Verify the file exists and has content
    ASSERT_TRUE(std::filesystem::exists(viz_filename + ".csv"));  // Add .csv extension
    
    
    // Verify CSV format
    auto lines = readTestFile();
    ASSERT_GT(lines.size(), 1);  // At least header + 1 data line
    
    auto header = parseCSVLine(lines[0]);
    ASSERT_EQ(header.size(), 6);  // Timestamp + 5 metrics
    ASSERT_EQ(header[0], "Timestamp");
    
    // Verify data lines are properly formatted
    for (size_t i = 1; i < lines.size(); ++i) {
        auto values = parseCSVLine(lines[i]);
        ASSERT_EQ(values.size(), 6);
        ASSERT_TRUE(isValidTimestamp(values[0]));
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}