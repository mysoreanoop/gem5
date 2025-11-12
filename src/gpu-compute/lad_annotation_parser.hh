/**
 * @file
 * The GPUDispatcher is the component of the shader that is responsible
 * for creating and dispatching WGs to the compute units. If all WGs in
 * a kernel cannot be dispatched simultaneously, then the dispatcher will
 * keep track of all pending WGs and dispatch them as resources become
 * available.
 */

#pragma once

#include "gem5/json.hpp"
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

class LADAnnotationParser {
public:
    /**
     * @brief Default constructor.
     */
    LADAnnotationParser(const std::string& _filename): filename(_filename) {};

    /**
     * @brief Loads and parses the dispatch information from a JSON file.
     * * This method reads the entire file, parses the JSON, and populates an
     * internal hash map for fast subsequent lookups. The existing data is cleared
     * before loading.
     * * @param filename The path to the JSON file.
     * @return true if the file was successfully loaded and parsed, false otherwise.
     */
    bool init();
    bool isDispLad(int dispatch_id) const;
    /**
     * @brief Retrieves a value from the first data collection (map<int, int>).
     * * @param dispatch_id The top-level dispatch ID.
     * @param key The key within the integer-to-integer map.
     * @return An std::optional<int> containing the value if found, otherwise std::nullopt.
     */
    std::optional<unsigned> getVgprTxln(int dispatch_id, int key) const;

    /**
     * @brief .
     * * @param dispatch_id The top-level dispatch ID.
     * @return .
     */
    std::vector<std::pair<int, int>> getMagicInsnMap(int dispatch_id) const;

    /**
     * @brief Retrieves a value from the third data collection (map<string, int>).
     * * @param dispatch_id The top-level dispatch ID.
     * @param key The key within the string-to-integer map.
     * @return An std::optional<int> containing the value if found, otherwise std::nullopt.
     */
    std::optional<int> get_kd_modifier(int dispatch_id, const std::string& key) const;

    /**
     * @brief Prints the entire contents of the dispatch database to std::cout.
     */
    void print_database() const;

private:
    const std::string& filename;

    // This struct holds the three required data collections for a single dispatch ID.
    struct DispatchData {
        std::unordered_map<unsigned, unsigned> vgpr_txn_map;
        std::vector<std::pair<int, int>> pc_insn_map;
        std::unordered_map<std::string, int> kd_map; // kernel descriptor overrides
    };

    // The main internal data structure. A hash map for fast O(1) average lookup by dispatch ID.
    std::unordered_map<int, DispatchData> dispatch_database_;
};
