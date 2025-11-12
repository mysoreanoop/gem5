#include "gpu-compute/lad_annotation_parser.hh"
#include <iomanip>  // For std::quoted
// --- Implementation ---

bool LADAnnotationParser::init() {
    std::ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return false;
    }

    try {
        json data = json::parse(file_stream);

        // Clear any old data to ensure a clean slate
        dispatch_database_.clear();

        // The top level of the JSON is expected to be an object where keys are dispatch IDs
        for (const auto& [dispatch_id_str, dispatch_data_json] : data.items()) {
            int dispatch_id = std::stoi(dispatch_id_str);
            DispatchData current_data;

            // 1. Populate the map of int -> int
            if (dispatch_data_json.contains("vgpr_txn_map")) {
                for (const auto& [key_str, value] : dispatch_data_json["vgpr_txn_map"].items()) {
                    current_data.vgpr_txn_map[(unsigned)(std::stoi(key_str))] = (unsigned)(value.get<int>());
                }
            }

            // 2. Populate the map of int -> string
            if (dispatch_data_json.contains("pc_insn_map")) {
                for (const auto& [key_str, value] : dispatch_data_json["pc_insn_map"].items()) {
                    current_data.pc_insn_map.push_back({std::stoi(key_str), value.get<int>()});
                }
            }

            // 3. Populate the map of string -> int
            if (dispatch_data_json.contains("kd_map")) {
                // nlohmann::json correctly handles direct conversion from a JSON object
                // to an STL map if the key/value types are compatible.
                current_data.kd_map = dispatch_data_json["kd_map"].get<std::unordered_map<std::string, int>>();
            }

            // Move the fully populated struct into our main database
            dispatch_database_[dispatch_id] = std::move(current_data);
        }
    } catch (const json::parse_error& e) {
        std::cerr << "Error: JSON parsing failed. " << e.what() << std::endl;
        return false;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: A dispatch ID or key in the JSON is not a valid integer. " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
        return false;
    }
    print_database();
    return true;
}

bool LADAnnotationParser::isDispLad(int dispatch_id) const {
    auto dispatch_it = dispatch_database_.find(dispatch_id);
    return (dispatch_it != dispatch_database_.end());
}

std::optional<unsigned> LADAnnotationParser::getVgprTxln(int dispatch_id, int key) const {
    // Find the dispatch ID
    auto dispatch_it = dispatch_database_.find(dispatch_id);
    if (dispatch_it == dispatch_database_.end()) {
        return std::nullopt; // Dispatch ID not found
    }

    // Find the key within the specific map
    const auto& vgpr_txn_map = dispatch_it->second.vgpr_txn_map;
    auto value_it = vgpr_txn_map.find(key);
    if (value_it == vgpr_txn_map.end()) {
        return std::nullopt; // Key not found
    }

    return value_it->second;
}

std::vector<std::pair<int, int>> LADAnnotationParser::getMagicInsnMap(int dispatch_id) const {
    auto dispatch_it = dispatch_database_.find(dispatch_id);
    if (dispatch_it == dispatch_database_.end()) {
        return std::vector<std::pair<int, int>>();
    }
    return dispatch_it->second.pc_insn_map;
}

std::optional<int> LADAnnotationParser::get_kd_modifier(int dispatch_id, const std::string& key) const {
    auto dispatch_it = dispatch_database_.find(dispatch_id);
    if (dispatch_it == dispatch_database_.end()) {
        return std::nullopt;
    }

    const auto& kd_map = dispatch_it->second.kd_map;
    auto value_it = kd_map.find(key);
    if (value_it == kd_map.end()) {
        return std::nullopt;
    }

    return value_it->second;
}

/**
 * @brief Prints the entire contents of the dispatch database to std::cout
 * in a formatted, human-readable way.
 */
void LADAnnotationParser::print_database() const {
    if (dispatch_database_.empty()) {
        std::cout << "LAD Annotation Database is empty." << std::endl;
        return;
    }

    // Iterate over each dispatch ID in the main database
    for (const auto& [dispatch_id, dispatch_data] : dispatch_database_) {
        std::cout << "========================================" << std::endl;
        std::cout << "--- Dispatch ID: " << dispatch_id << " ---" << std::endl;
        std::cout << "========================================" << std::endl;

        // 1. Print vgpr_txn_map (int -> int)
        std::cout << "\n  [VGPR TXN Map (int -> int)]:" << std::endl;
        if (dispatch_data.vgpr_txn_map.empty()) {
            std::cout << "    (empty)" << std::endl;
        } else {
            for (const auto& [key, value] : dispatch_data.vgpr_txn_map) {
                std::cout << "    " << key << " -> " << value << std::endl;
            }
        }

        // 2. Print pc_insn_map (int -> string)
        std::cout << "\n  [PC INSN Map (int -> string)]:" << std::endl;
        if (dispatch_data.pc_insn_map.empty()) {
            std::cout << "    (empty)" << std::endl;
        } else {
            for (const auto& p : dispatch_data.pc_insn_map) {
                // Use std::quoted to handle any special characters in the string
                std::cout << "    " << p.first << " -> " << p.second << std::endl;
            }
        }

        // 3. Print kd_map (string -> int)
        std::cout << "\n  [KD Map (string -> int)]:" << std::endl;
        if (dispatch_data.kd_map.empty()) {
            std::cout << "    (empty)" << std::endl;
        } else {
            for (const auto& [key, value] : dispatch_data.kd_map) {
                std::cout << "    " << std::quoted(key) << " -> " << value << std::endl;
            }
        }
        std::cout << std::endl; // Add space before the next dispatch ID
    }
}
