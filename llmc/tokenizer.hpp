#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "utils.h"

class Tokenizer {
private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, uint32_t> token_to_id_;
    uint32_t vocab_size_;
    uint32_t eot_token_;
    bool init_ok_;

     // Helper for longest token match
    std::pair<uint32_t, size_t> find_longest_token(const std::string& text, size_t start) const {
        size_t longest_len = 0;
        uint32_t token_id = UINT32_MAX;
        
        // Try each possible substring starting at 'start'
        for (size_t end = start + 1; end <= text.length(); end++) {
            std::string substr = text.substr(start, end - start);
            auto it = token_to_id_.find(substr);
            if (it != token_to_id_.end() && substr.length() > longest_len) {
                longest_len = substr.length();
                token_id = it->second;
            }
        }
        
        return {token_id, longest_len};
    }

public:
    Tokenizer() : init_ok_(false) {}

    void init(const std::string& dict_file) {
        FILE* file = fopen(dict_file.c_str(), "r");
        if (!file) {
            fprintf(stderr, "Failed to open dictionary file: %s\n", dict_file.c_str());
            return;
        }

        char line[1024];
        uint32_t token_id = 0;
        tokens_.clear();
        token_to_id_.clear();

        while (fgets(line, sizeof(line), file)) {
            size_t len = strlen(line);
            if (len > 0 && line[len-1] == '\n') {
                line[len-1] = '\0';
            }

            char* token = strtok(line, "\t");
            char* id_str = strtok(NULL, "\t");
            
            if (id_str) {
                token_id = std::stoul(id_str);
            }

            // Ensure vector has space
            if (token_id >= tokens_.size()) {
                tokens_.resize(token_id + 1);
            }
            
            tokens_[token_id] = token;
            token_to_id_[token] = token_id;
            
            if (!id_str) {
                token_id++;
            }
        }

        vocab_size_ = tokens_.size();
        eot_token_ = token_to_id_["<|endoftext|>"];
        init_ok_ = true;

        fcloseCheck(file);
    }

    std::vector<uint32_t> encode_string(const std::string& text) const {
        std::vector<uint32_t> result;
        size_t pos = 0;
        
        while (pos < text.length()) {
            auto [token_id, length] = find_longest_token(text, pos);
            if (token_id == UINT32_MAX || length == 0) {
                // No token found, move forward by one byte
                pos++;
            } else {
                result.push_back(token_id);
                pos += length;
            }
        }
        
        return result;
    }

    std::string decode_string(const std::vector<uint32_t>& token_ids) const {
        std::string result;
        for (uint32_t id : token_ids) {
            if (id < vocab_size_) {
                result += tokens_[id];
            }
        }
        return result;
    }


    uint32_t encode(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        fprintf(stderr, "Token not found: %s\n", token.c_str());
        return UINT32_MAX;
    }

    std::string decode(uint32_t token_id) const {
        if (!init_ok_) {
            fprintf(stderr, "Tokenizer not initialized\n");
            return "";
        }
        if (token_id >= vocab_size_) {
            fprintf(stderr, "Invalid token ID: %u\n", token_id);
            return "";
        }
        return tokens_[token_id];
    }

    uint32_t get_vocab_size() const { return vocab_size_; }
    uint32_t get_eot_token() const { return eot_token_; }
    bool is_initialized() const { return init_ok_; }
};

#endif